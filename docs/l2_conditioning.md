# Astra 在线 L3＋L2 条件

新 `astra.prepare` / `scripts/prepare_core9.py` 产物默认使用 `l3_l2`。旧 manifest 缺少该字段时仍按 `pure_l3` 解释；修改本仓库不会升级旧运行的冻结源码。启用新评测前需要独立 GPU smoke rollout，不在已有场景中改模型模式。

## 执行会话

1. 用 `inspect_live.py WORKER` 读取待审请求，实际查看示范和当前多视角图像；框选前另看原始 `cam_head` 图（若请求键为 `cam_high` 则使用该键），不要在联系图上取坐标。
2. 首次状态为 `PREPARE_CONDITIONING`，尚未调用模型推理。保存条件 JSON，例如：

```json
{
  "subgoal": "grasp",
  "l2_instruction": "Grasp the charger.",
  "cue_rgb_path": "/absolute/path/from/current/request/000000_cam_head.png",
  "l2_bboxes_xyxy": [[120, 80, 180, 140]],
  "box_source": "astra_visual",
  "evidence": "Describe the observed target and why this box identifies it.",
  "change_reason": "Initial visually reviewed grasp stage"
}
```

以上框仅为格式示例，不能直接用于真实场景。`subgoal` 应匹配实际已审阅示范阶段；`general_pickup` 继续采用已有用户豁免。框来源允许 `astra_visual` 或 `validated_visual_locator`。原图不画框，坐标为原图像素；只有明确不需框的操作可传 `[]` 并写 `unboxed_reason`。

```bash
python3 scripts/respond_live.py WORKER REQUEST_ID condition 0 '已审阅当前目标' \
  --segment grasp --scene-review '当前场景与示范的差异' \
  --conditioning-file WORKER/conditioning_input.json
```

3. 控制桥验证条件并冻结 cue，随后调用模型。新的动作请求仍须看图审查；新运行默认每 15 次环境动作审查一次，low policy、修正和接管使用相同间隔。已知风险可缩短本批，用 `--short-batch-reason`（或决策 `reason`）记录依据；chunk 剩余不足 15 步时提前审查。`condition` 自身不执行机器人动作。控制桥逐动作检查合法性和官方终态，`replan_steps=16` 不变。manifest、request 和 summary 保存实际监控间隔，decision 的 `monitoring_batch` 保存所选步数和缩短原因。历史缺字段 manifest 仍按原 4 步解释。
4. 同一子目标保持起始 cue/框不变，同时更新实时观测。普通重规划明确复用 `request.conditioning.conditioning_version`：

```bash
python3 scripts/respond_live.py WORKER REQUEST_ID replan 0 '依据当前观察重新推理' \
  --segment grasp --conditioning-version EPISODE:c1
```

切换目标或需要重建恢复阶段时，基于待审请求的新原图提交新的条件 JSON，并填写 `change_reason`；使用 `replan --conditioning-file FILE`。接管期间不用 low policy；交还使用 `hand_back`，附 `--recovery-evidence TEXT` 及新条件文件或仍有效的条件版本。不能用 `--instruction` 替换整局 L3，也不能只换 `--segment` 就继续批准旧动作。

## 审计与故障

- episode 的 `conditioning/` 保存每版 cue PNG、哈希、坐标、指令和起始 step。
- episode 的 `inference_conditions.jsonl` 关联推理 ID、待审 request ID、完整条件和模型实际输入回执。
- 每局 `runs/layout_N/model_conditioning.jsonl` 是模型侧日志；同路径加 `.receipt.json` 是原子发布的最新回执。原生 WebSocket `update_obs` 在客户端暂存观测，`get_action` 实际发出 infer；服务端执行 update_obs/get_action。无需修改共享传输协议。
- 回执验证的是模型适配器输入及推理是否完成，不表示测量了模型 attention。图像预处理、框缩放和绘制仍走真实 ImageWAM 服务代码。
- 非法条件被拒后场景保持不动，可修正 JSON 继续提交。接口故障进入 `paused_for_main`，Monitor 给出告警；主会话先修复，再运行 `resume_conditioning.py --session WORKER --pause-id ID --reason TEXT`，旧 pause ID 不能恢复新故障。没有自动降级、自动批准或 reset。
- Astra 提供条件属于高层辅助；即使动作修正/接管均为 0，成功也归为辅助成功。

## 验证范围

测试覆盖首次推理门控、非法框/错图拒绝、跨 chunk 固定 cue、阶段切换、接管及交还、回执不匹配暂停恢复、独占响应发布、历史 pure L3 兼容，以及真实 ImageWAM 适配器的 CPU 输入路径。CPU 测试用假 policy 代替权重推理，不能替代 GPU 仿真验证。
