"""Read-only Codex token accounting with explicit episode timestamp attribution."""
import argparse
import bisect
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

FIELDS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens')
ZERO = dict.fromkeys(FIELDS, 0)


def subtract(a, b):
    d = {k: a.get(k, 0)-b.get(k, 0) for k in FIELDS}
    assert all(v >= 0 for v in d.values()), 'Cumulative usage decreased; manual reconciliation required'
    assert d['total_tokens'] == d['input_tokens']+d['output_tokens']
    d['uncached_input_tokens'] = d['input_tokens']-d['cached_input_tokens']
    d['uncached_input_plus_output_tokens'] = d['uncached_input_tokens']+d['output_tokens']
    return d


def events(path, cutoff):
    result=[]
    for line in path.open():
        try:
            row=json.loads(line)
        except ValueError:
            continue  # Live file may end in an incomplete line.
        p=row.get('payload', {})
        if row.get('type') != 'event_msg' or p.get('type') != 'token_count' or not p.get('info'):
            continue
        ts=datetime.fromisoformat(row['timestamp'].replace('Z','+00:00')).timestamp()
        if ts > cutoff:
            continue
        usage=p['info']['total_token_usage']
        subtract(usage, result[-1][1] if result else ZERO)
        result.append((ts, usage))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--codex-sessions', type=Path, default=Path('/share-2/home/vla/.codex-wangzihang/sessions'))
    args=parser.parse_args()
    batch=args.session.resolve(); cutoff=time.time()
    rows=[]; workers=[]; sources=[]
    for worker in sorted((batch/'workers').iterdir()):
        status=worker/'supervisor_status.json'
        if not status.exists():
            workers.append(dict(task=worker.name, started=False))
            continue
        status=json.loads(status.read_text()); tid=status.get('thread_id')
        paths=list(args.codex_sessions.rglob('*'+str(tid)+'*.jsonl'))
        if len(paths)!=1:
            raise ValueError(f'{worker.name}: expected one raw session log, found {paths}')
        ev=events(paths[0],cutoff)
        if not ev:
            raise ValueError(f'{worker.name}: no token usage')
        sources.append(str(paths[0])); times=[x[0] for x in ev]
        summaries=sorted(worker.glob('*/seed_*/*/summary.json'), key=lambda f:f.stat().st_mtime)
        previous=ZERO; completed=0
        for f in summaries:
            end=f.stat().st_mtime
            if end>cutoff:
                continue
            summary=json.loads(f.read_text()); idx=bisect.bisect_right(times,end)-1
            cumulative=ev[idx][1] if idx>=0 else ZERO
            usage=subtract(cumulative,previous)
            after=ev[idx+1] if idx+1<len(ev) else None
            uncertainty=subtract(after[1], cumulative) if after else None
            rows.append(dict(task=status['task'],episode=f.parent.name,status=summary['status'],
                official_success=summary.get('official_success'),official_terminal=summary.get('official_terminal'),
                steps=summary['steps'],agent_decisions=summary.get('agent_decisions'),
                summary_path=str(f),end_time_source='summary.json mtime',end_timestamp=end,
                usage_boundary_timestamp=ev[idx][0] if idx>=0 else None,
                next_usage_timestamp=after[0] if after else None,
                boundary_next_update_delta=uncertainty,
                attribution='Cumulative usage through last token event at/before summary creation, minus previous episode boundary; first episode includes demo/startup supervision',
                **usage))
            completed+=1;previous=cumulative
        cumulative=subtract(ev[-1][1],ZERO)
        workers.append(dict(task=status['task'],started=True,thread_id=tid,raw_log=str(paths[0]),
            token_snapshot_timestamp=ev[-1][0],completed_episodes=completed,
            task_three_episodes_complete=completed==3,**cumulative,
            ongoing_and_post_episode_usage=subtract(ev[-1][1],previous)))
    def aggregate(items):
        return dict(count=len(items),totals={k:sum(x[k] for x in items) for k in (*FIELDS,'uncached_input_tokens','uncached_input_plus_output_tokens')},
                    means={k:statistics.mean(x[k] for x in items) if items else None for k in (*FIELDS,'uncached_input_tokens','uncached_input_plus_output_tokens')})
    active=[w for w in workers if w['started']]
    result=dict(as_of_timestamp=cutoff,as_of_local=datetime.fromtimestamp(cutoff,ZoneInfo('Asia/Shanghai')).isoformat(),
                scope='Only task execution Codex supervisors in this batch. Excludes main session, demo preparation/development agents, earlier sessions and ImageWAM GPU inference.',
                methodology='Use total_token_usage once per boundary; never sum cumulative snapshots or add reasoning to output. Episode attribution uses timestamp bins, not exact billing tags; bracket deltas retained. Uncached input plus output is a descriptive sum, not billed/equivalent tokens.',
                warnings=['Ongoing episodes excluded from finished-episode mean. No completed three-episode task mean until all three are terminal.',
                          'Finished sample mixes finite-recovery early terminations and episodes continuing after unlimited-recovery authorization; not a representative final benchmark estimate.'],
                workers=workers,episodes=rows,sources=sources,
                completed_episode_statistics=aggregate(rows),
                successful_episode_statistics=aggregate([r for r in rows if r['official_success']]),
                completed_three_episode_task_statistics=aggregate([w for w in active if w['task_three_episodes_complete']]),
                all_started_supervisor_statistics=aggregate(active))
    # Independent conservation check: completed bins + unfinished remainder equal
    # the cumulative supervisor totals without double counting cached/reasoning.
    for k in FIELDS:
        assert sum(r[k] for r in rows)+sum(w['ongoing_and_post_episode_usage'][k] for w in active)==sum(w[k] for w in active)
    (batch/'token_usage_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (batch/'token_usage_episodes.csv').open('w',newline='') as stream:
        columns=['task','episode','status','official_success','steps',*FIELDS,'uncached_input_tokens','uncached_input_plus_output_tokens']
        writer=csv.DictWriter(stream,columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Codex 任务监督 token 统计','',f'快照时间：{result["as_of_local"]}',
           '', '仅本轮独立任务监督会话；不含主会话、示范准备/开发会话、旧批次和 ImageWAM GPU 推理。',
           '累计 total=input+output；缓存输入是 input 的子集，推理是 output 的子集，不重复相加。',
           '按 summary.json 创建时间切分累计 token；第一局包括示范审阅和启动等待。切分处可能有一次 usage 更新的归属误差，JSON 保存前后边界。',
           '', '| 已结束单局 | 结果 | 步数 | 总 token（含缓存） | 缓存输入 | 非缓存输入 | 输出（含推理） | 非缓存输入+输出 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f'| {r["task"]} / {r["episode"]} | {r["status"]} | {r["steps"]} | '+ ' | '.join(f'{r[k]:,}' for k in ['total_tokens','cached_input_tokens','uncached_input_tokens','output_tokens','uncached_input_plus_output_tokens'])+' |')
    mean=result['completed_episode_statistics']['means'];success=result['successful_episode_statistics']['means']
    lines+=['',f'已结束 {len(rows)} 局均值：总 token {mean["total_tokens"]:,.0f}；非缓存输入+输出 {mean["uncached_input_plus_output_tokens"]:,.0f}。' if rows else '尚无已结束局。',
            f'成功 {result["successful_episode_statistics"]["count"]} 局均值：总 token {success["total_tokens"]:,.0f}；非缓存输入+输出 {success["uncached_input_plus_output_tokens"]:,.0f}。' if result['successful_episode_statistics']['count'] else '尚无成功局。',
            '', '样本含旧次数限制下提前结束的失败局，与无限恢复后的成功局；不能当作未来完整测试的保证。非缓存输入+输出不是计费折算值。',
            f'完成全部三局的任务数：{result["completed_three_episode_task_statistics"]["count"]}。',
            '', '| 任务会话 | 已结束局数 | 截至目前总 token | 非缓存输入+输出 |', '| --- | --- | --- | --- |']
    for w in active:lines.append(f'| {w["task"]} | {w["completed_episodes"]}/3 | {w["total_tokens"]:,} | {w["uncached_input_plus_output_tokens"]:,} |')
    lines+=['','这是定时点快照，运行中部分仍会增加。不能把所有会话累计总量除以已结束局数当作单局均值。',
            '字段说明：[OpenAI reasoning 文档](https://developers.openai.com/api/docs/guides/reasoning)、[Prompt caching 文档](https://developers.openai.com/api/docs/guides/prompt-caching)。']
    (batch/'token_usage_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['as_of_local','completed_episode_statistics','successful_episode_statistics','completed_three_episode_task_statistics','all_started_supervisor_statistics']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
