# RoboDojo 训练数据 L2 指令清单

提取日期：2026-09-22。以下文本直接提取自指定启动目录关联的训练 manifest，保留英文原文。

## 来源

- 用户指定的启动目录只有 `launch_receipt.json`；本清单依据其中记录的 manifest 路径定位，不据此判断该次训练是否实际完成。
- 训练 manifest：[robodojo_l2_bbox_v2.json](/share-2/code/liupeiqi/imagewam-robodojo-9b-l3mix-launch-08ff23e8/experiments/robodojo_l2_bbox/robodojo_l2_bbox_v2.json)
- SHA256：`991598b7ca2b669b6474f0d5855e24c318d4d12dd5a20182b6f677f61b53eb5c`，与启动回执一致。
- 原始标注目录：`/share-2/code/chenqi/robodojo/final_clean_20260902/episodes/`。
- 常规 L2 来源字段：`segments[].annotation.canonical_label`；训练 manifest 中为每个 segment 的 `instruction`。
- L3 任务文本：[tasks.jsonl](/share-2/home/vla/yulonglu/data/RoboDojo/RoboDojo_lerobot_v21_video/meta/tasks.jsonl)
- 实际统计：3,500 个 episode、35 类任务、33,929 个阶段、848 条全局去重指令。

## 文本与阶段格式

L2 通常是描述当前操作的英文短句，包含动作、对象、目标位置，部分明确左右臂、颜色、大小或顺序，也有观察、等待和撤臂阶段。

L3+L2 分支的文本拼接格式：

```text
<L3 总任务指令>
Current subtask instruction: <当前阶段的 L2 指令>
```

每条阶段记录的格式为 `[start_frame, instruction, annotation_required, boxes]`。清单声明相机为 `cam_high`，图像尺寸为 `[480, 640]`（高、宽），框字段为 `[x_min, y_min, x_max, y_max]`。

**例外：task 32** 在训练 manifest 中使用 L3 全任务文字作为 L2，整局一个阶段；原始标注虽然有合并的 canonical_label，但不能把该合并文字当作实际训练使用的 L2。

## 完整阶段示例

### Episode 1000 / Task 10

L3：Hang all the mugs on the mug rack.

| 起始帧 | L2 原文 | 需要框 |
| ---: | --- | --- |
| 0 | pick up the yellow mug | 是 |
| 84 | hang the yellow mug on the mug rack | 是 |
| 180 | pick up the white mug | 是 |
| 248 | hang the white mug on the mug rack | 是 |
| 341 | pick up the purple mug | 是 |
| 401 | hang the purple mug on the mug rack | 是 |
| 500 | retract the left arm | 否 |

### Episode 2200 / Task 22

L3：Plug the charger into the power strip.

| 起始帧 | L2 原文 | 需要框 |
| ---: | --- | --- |
| 0 | pick up the charger with the right arm | 是 |
| 33 | handover the charger from the right arm to the left arm | 是 |
| 122 | insert the charger into the power strip | 是 |
| 174 | retract the left arm | 否 |

### Episode 2300 / Task 23

L3：Pour all the balls from the cup into the vase.

| 起始帧 | L2 原文 | 需要框 |
| ---: | --- | --- |
| 0 | pick up the cup with the right arm | 是 |
| 69 | handover the cup from the right arm to the left arm | 是 |
| 195 | move the cup to the vase and pour all balls into it | 是 |
| 264 | return the empty cup and place it on the table | 是 |
| 320 | retract the left arm | 否 |

## 按任务列出全部指令

各任务内按第一次出现的顺序去重。括号列出一个可定位的 episode 和起始帧。不同任务可能共用同一文本，因此各节条目数之和可能大于全局去重数。

### Task 00（94 条）

L3：Arrange the numbers from left to right to form the largest possible number, and place them on the pad.

- `pick up 6 with the left arm`（episode 0，起始帧 0）
- `place 6 on the first pad with the left arm`（episode 0，起始帧 45）
- `pick up 5 with the right arm`（episode 0，起始帧 89）
- `place 5 in the center with the right arm`（episode 0，起始帧 144）
- `pick up 5 from the center with the left arm`（episode 0，起始帧 184）
- `place 5 on the second pad with the left arm`（episode 0，起始帧 229）
- `pick up 2 with the right arm`（episode 0，起始帧 263）
- `place 2 on the third pad with the right arm`（episode 0，起始帧 321）
- `pick up 0 with the left arm`（episode 0，起始帧 362）
- `place 0 in the center with the left arm`（episode 0，起始帧 417）
- `pick up 0 from the center with the right arm`（episode 0，起始帧 465）
- `place 0 on the fourth pad with the right arm`（episode 0，起始帧 518）
- `retract right arm`（episode 0，起始帧 546）
- `pick up 8 with the left arm`（episode 1，起始帧 0）
- `place 8 on the first pad with the left arm`（episode 1，起始帧 42）
- `pick up 6 with the right arm`（episode 1，起始帧 81）
- `place 6 in the center with the right arm`（episode 1，起始帧 135）
- `pick up 6 from the center with the left arm`（episode 1，起始帧 180）
- `place 6 on the second pad with the left arm`（episode 1，起始帧 232）
- `place 5 on the third pad with the right arm`（episode 1，起始帧 316）
- `pick up 3 with the left arm`（episode 1，起始帧 355）
- `place 3 in the center with the left arm`（episode 1，起始帧 410）
- `pick up 3 from the center with the right arm`（episode 1，起始帧 448）
- `place 3 on the fourth pad with the right arm`（episode 1，起始帧 493）
- `pick up 9 with the right arm`（episode 2，起始帧 0）
- `place 9 in the center with the right arm`（episode 2，起始帧 42）
- `pick up 9 from the center with the left arm`（episode 2，起始帧 84）
- `place 9 on the first pad with the left arm`（episode 2，起始帧 132）
- `pick up 4 with the right arm`（episode 2，起始帧 165）
- `place 4 in the center with the right arm`（episode 2，起始帧 220）
- `pick up 4 from the center with the left arm`（episode 2，起始帧 261）
- `place 4 on the second pad with the left arm`（episode 2，起始帧 309）
- `pick up 3 with the right arm`（episode 2，起始帧 341）
- `place 3 on the third pad with the right arm`（episode 2，起始帧 397）
- `pick up 2 with the left arm`（episode 2，起始帧 438）
- `place 2 in the center with the left arm`（episode 2，起始帧 494）
- `pick up 2 from the center with the right arm`（episode 2，起始帧 534）
- `place 2 on the fourth pad with the right arm`（episode 2，起始帧 587）
- `pick up 8 with the right arm`（episode 3，起始帧 170）
- `place 8 in the center with the right arm`（episode 3，起始帧 223）
- `pick up 8 from the center with the left arm`（episode 3，起始帧 267）
- `place 8 on the second pad with the left arm`（episode 3，起始帧 313）
- `pick up 4 with the left arm`（episode 3，起始帧 348）
- `place 4 in the center with the left arm`（episode 3，起始帧 393）
- `pick up 4 from the center with the right arm`（episode 3，起始帧 439）
- `place 4 on the third pad with the right arm`（episode 3，起始帧 485）
- `place 2 on the fifth pad with the right arm`（episode 3，起始帧 844）
- `retract both arms`（episode 3，起始帧 882）
- `place 5 on the first pad with the left arm`（episode 4，起始帧 132）
- `pick up 0 with the right arm`（episode 4，起始帧 602）
- `place 0 on the fifth pad with the right arm`（episode 4，起始帧 644）
- `pick up 7 with the left arm`（episode 6，起始帧 0）
- `place 7 on the first pad with the left arm`（episode 6，起始帧 45）
- `pick up 1 with the left arm`（episode 6，起始帧 532）
- `place 1 in the center with the left arm`（episode 6，起始帧 589）
- `pick up 1 from the center with the right arm`（episode 6，起始帧 626）
- `place 1 on the fifth pad with the right arm`（episode 6，起始帧 672）
- `pick up 1 with the right arm`（episode 7，起始帧 424）
- `place 1 on the fourth pad with the right arm`（episode 7，起始帧 471）
- `place 1 on the third pad with the right arm`（episode 8，起始帧 305）
- `place 7 on the second pad with the left arm`（episode 9，起始帧 131）
- `place 6 on the third pad with the right arm`（episode 9，起始帧 226）
- `pick up 9 with the left arm`（episode 11，起始帧 0）
- `pick up 7 with the right arm`（episode 11，起始帧 80）
- `place 7 in the center with the right arm`（episode 11，起始帧 136）
- `pick up 7 from the center with the left arm`（episode 11，起始帧 172）
- `pick up 7 from the center with the right arm`（episode 13，起始帧 0）
- `pick up 1 from the center with the left arm`（episode 13，起始帧 515）
- `place 2 on the second pad with the left arm`（episode 15，起始帧 214）
- `place 6 in the center with the left arm`（episode 16，起始帧 292）
- `pick up 6 from the center with the right arm`（episode 16，起始帧 333）
- `place 5 on the fourth pad with the right arm`（episode 17，起始帧 411）
- `place 4 on the fifth pad with the right arm`（episode 17，起始帧 491）
- `place 3 on the second pad with the left arm`（episode 21，起始帧 126）
- `place 7 in the center with the left arm`（episode 24，起始帧 208）
- `place 7 on the third pad with the right arm`（episode 24，起始帧 294）
- `pick up 5 with the left arm`（episode 27，起始帧 162）
- `place 4 on the second pad with the right arm`（episode 29，起始帧 135）
- `place 4 on the fourth pad with the right arm`（episode 33，起始帧 457）
- `place 6 on the third pad with the left arm`（episode 36，起始帧 407）
- `place 2 on the third pad with the left arm`（episode 40，起始帧 294）
- `place 5 in the center with the left arm`（episode 47，起始帧 391）
- `pick up 5 from the center with the right arm`（episode 47，起始帧 429）
- `place 5 on the third pad with the left arm`（episode 52，起始帧 394）
- `place 3 on the fifth pad with the right arm`（episode 52，起始帧 734）
- `pick up 9 from the first pad with the left arm`（episode 56，起始帧 359）
- `place 3 on the third pad with the left arm`（episode 59，起始帧 215）
- `place 6 on the fourth pad with the right arm`（episode 63，起始帧 388）
- `place 2 in the center with the right arm`（episode 83，起始帧 308）
- `pick up 2 from the center with the left arm`（episode 83，起始帧 349）
- `place 4 on the third pad with the left arm`（episode 89，起始帧 397）
- `place 7 on the third pad with the left arm`（episode 90，起始帧 213）
- `place 5 on the table with the left arm`（episode 90，起始帧 411）
- `place 5 on the fifth pad with the right arm`（episode 90，起始帧 492）

### Task 01（18 条）

L3：Build a tower using the wooden blocks and wooden boards.

- `pick up the first base block`（episode 100，起始帧 0）
- `place the first base block on the left base position`（episode 100，起始帧 33）
- `pick up the second base block`（episode 100，起始帧 72）
- `place the second base block on the right base position`（episode 100，起始帧 117）
- `pick up the long wooden board`（episode 100，起始帧 160）
- `place the long wooden board on the base blocks`（episode 100，起始帧 207）
- `pick up the first upper block`（episode 100，起始帧 249）
- `place the first upper block on the left position on long wooden board`（episode 100，起始帧 295）
- `pick up the second upper block`（episode 100，起始帧 340）
- `place the second upper block on the right position on long wooden board`（episode 100，起始帧 382）
- `pick up the short wooden board`（episode 100，起始帧 427）
- `place the short wooden board on the upper blocks`（episode 100，起始帧 470）
- `pick up the brown wooden block`（episode 100，起始帧 516）
- `place the brown wooden block on the short wooden board`（episode 100，起始帧 559）
- `pick up the green block`（episode 100，起始帧 604）
- `place the green block on the brown wooden block`（episode 100，起始帧 648）
- `retract the left arm`（episode 100，起始帧 692）
- `retract the right arm`（episode 101，起始帧 696）

### Task 02（65 条）

L3：Sort the objects by category into the three baskets.

- `pick up the camera with the left arm`（episode 200，起始帧 0）
- `place the camera in the white basket with the left arm`（episode 200，起始帧 47）
- `pick up the camera with the right arm`（episode 200，起始帧 207）
- `place the camera in the center for handover with the right arm`（episode 200，起始帧 266）
- `pick up the camera from the center with the left arm`（episode 200，起始帧 303）
- `pick up the toy car with the left arm`（episode 200，起始帧 417）
- `place the toy car in the blue basket with the left arm`（episode 200，起始帧 466）
- `pick up the pen with the right arm`（episode 200，起始帧 521）
- `place the pen in the red basket with the right arm`（episode 200，起始帧 578）
- `retract both arms`（episode 200，起始帧 726）
- `pick up the pen with the left arm`（episode 201，起始帧 0）
- `place the pen in the white basket with the left arm`（episode 201，起始帧 81）
- `place the camera in the blue basket with the right arm`（episode 201，起始帧 330）
- `pick up the watch with the right arm`（episode 201，起始帧 484）
- `place the watch in the red basket with the right arm`（episode 201，起始帧 546）
- `pick up the rabbit plush toy with the left arm`（episode 202，起始帧 0）
- `place the rabbit plush toy in the white basket with the left arm`（episode 202，起始帧 62）
- `pick up the rabbit plush toy with the right arm`（episode 202，起始帧 289）
- `place the rabbit plush toy in the white basket with the right arm`（episode 202，起始帧 330）
- `place the camera in the blue basket with the left arm`（episode 202，起始帧 588）
- `pick up the watch with the left arm`（episode 203，起始帧 0）
- `place the watch in the white basket with the left arm`（episode 203，起始帧 83）
- `place the pen in the blue basket with the left arm`（episode 204，起始帧 257）
- `place the pen in the blue basket with the right arm`（episode 204，起始帧 519）
- `pick up the ring with the right arm`（episode 204，起始帧 560）
- `place the ring in the red basket with the right arm`（episode 204，起始帧 627）
- `pick up the toy tank with the left arm`（episode 205，起始帧 0）
- `place the toy tank in the white basket with the left arm`（episode 205，起始帧 78）
- `place the watch in the blue basket with the left arm`（episode 205，起始帧 447）
- `place the watch in the blue basket with the right arm`（episode 205，起始帧 600）
- `place the camera in the red basket with the right arm`（episode 205，起始帧 858）
- `place the watch in the center for handover with the left arm`（episode 208，起始帧 463）
- `pick up the watch from the center with the right arm`（episode 208，起始帧 514）
- `pick up the toy car with the right arm`（episode 210，起始帧 327）
- `place the toy car in the blue basket with the right arm`（episode 210，起始帧 404）
- `place the rabbit plush toy in the blue basket with the left arm`（episode 211，起始帧 289）
- `place the rabbit plush toy in the blue basket with the right arm`（episode 211，起始帧 417）
- `place the watch in the center for handover with the right arm`（episode 212，起始帧 200）
- `pick up the watch from the center with the left arm`（episode 212，起始帧 245）
- `place the toy car in the center for handover with the left arm`（episode 212，起始帧 656）
- `pick up the toy car from the center with the right arm`（episode 212，起始帧 690）
- `place the toy car in the red basket with the right arm`（episode 212，起始帧 743）
- `place the toy tank in the blue basket with the left arm`（episode 213，起始帧 225）
- `place the rabbit plush toy in the red basket with the right arm`（episode 213，起始帧 407）
- `pick up the yellow block with the left arm`（episode 217，起始帧 212）
- `place the yellow block in the blue basket with the left arm`（episode 217，起始帧 336）
- `place the toy car in the white basket with the left arm`（episode 222，起始帧 46）
- `place the pen in the center for handover with the right arm`（episode 224，起始帧 220）
- `pick up the pen from the center with the left arm`（episode 224，起始帧 279）
- `place the toy car in the center for handover with the right arm`（episode 237，起始帧 369）
- `pick up the toy car from the center with the left arm`（episode 237，起始帧 423）
- `place the pen in the center for handover with the left arm`（episode 242，起始帧 535）
- `pick up the pen from the center with the right arm`（episode 242，起始帧 593）
- `pick up the toy tank with the right arm`（episode 249，起始帧 391）
- `place the toy tank in the blue basket with the right arm`（episode 249，起始帧 435）
- `place the watch in the red basket with the left arm`（episode 258，起始帧 308）
- `place the watch in the white basket with the right arm`（episode 266，起始帧 132）
- `place the toy tank in the center for handover with the right arm`（episode 267，起始帧 201）
- `pick up the toy tank from the center with the left arm`（episode 267，起始帧 249）
- `place the camera in the white basket with the right arm`（episode 269，起始帧 125）
- `place the toy tank in the red basket with the right arm`（episode 286，起始帧 540）
- `place the rabbit plush toy in the center for handover with the right arm`（episode 291，起始帧 117）
- `pick up the rabbit plush toy from the center with the left arm`（episode 291，起始帧 157）
- `place the camera in the center for handover with the left arm`（episode 299，起始帧 605）
- `pick up the camera from the center with the right arm`（episode 299，起始帧 648）

### Task 03（26 条）

L3：Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue.

- `pick up the left cup`（episode 300，起始帧 0）
- `place the left cup over the green block`（episode 300，起始帧 32）
- `pick up the middle cup`（episode 300，起始帧 73）
- `place the middle cup over the blue block`（episode 300，起始帧 108）
- `pick up the right cup`（episode 300，起始帧 152）
- `place the right cup over the red block`（episode 300，起始帧 201）
- `pick up the right cup from the red block`（episode 300，起始帧 271）
- `return the right cup to the right position`（episode 300，起始帧 306）
- `pick up the left cup from the green block`（episode 300，起始帧 346）
- `return the left cup to the left position`（episode 300，起始帧 388）
- `pick up the middle cup from the blue block`（episode 300，起始帧 429）
- `return the middle cup to the middle position`（episode 300，起始帧 479）
- `retract the right arm`（episode 300，起始帧 523）
- `place the left cup over the red block`（episode 301，起始帧 32）
- `place the middle cup over the green block`（episode 301，起始帧 107）
- `place the right cup over the blue block`（episode 301，起始帧 200）
- `pick up the left cup from the red block`（episode 301，起始帧 270）
- `pick up the middle cup from the green block`（episode 301，起始帧 345）
- `pick up the right cup from the blue block`（episode 301，起始帧 427）
- `place the left cup over the blue block`（episode 302，起始帧 32）
- `pick up the left cup from the blue block`（episode 302，起始帧 435）
- `retract the left arm`（episode 302，起始帧 510）
- `place the middle cup over the red block`（episode 303，起始帧 108）
- `place the right cup over the green block`（episode 303，起始帧 200）
- `pick up the middle cup from the red block`（episode 303，起始帧 270）
- `pick up the right cup from the green block`（episode 303，起始帧 352）

### Task 04（8 条）

L3：Pick up the coin from the holder and insert it precisely into the coin bank.

- `pick up the coin from the holder`（episode 400，起始帧 0）
- `hand over the coin from the right arm to the left arm`（episode 400，起始帧 43）
- `insert the coin into the coin bank`（episode 400，起始帧 124）
- `retract the left arm after inserting the coin`（episode 400，起始帧 170）
- `hand over the coin from the left arm to the right arm`（episode 404，起始帧 45）
- `retract the right arm after inserting the coin`（episode 404，起始帧 175）
- `insert the coin into the coin bank and retract the left arm`（episode 462，起始帧 128）
- `insert the coin into the coin bank and retract the right arm`（episode 476，起始帧 128）

### Task 05（14 条）

L3：Arrange the letters to spell "RoboDojo" in a row.

- `pick up R`（episode 500，起始帧 0）
- `place R in the first position`（episode 500，起始帧 35）
- `pick up o`（episode 500，起始帧 72）
- `place o in the second position`（episode 500，起始帧 106）
- `pick up b`（episode 500，起始帧 145）
- `place b in the third position`（episode 500，起始帧 182）
- `place o in the fourth position`（episode 500，起始帧 257）
- `pick up D`（episode 500，起始帧 303）
- `place D in the fifth position`（episode 500，起始帧 352）
- `place o in the sixth position`（episode 500，起始帧 432）
- `pick up j`（episode 500，起始帧 473）
- `place j in the seventh position`（episode 500，起始帧 509）
- `place o in the eighth position`（episode 500，起始帧 584）
- `retract right arm`（episode 500，起始帧 624）

### Task 06（52 条）

L3：Insert and tighten each screw into the nut of the same color.

- `pick up the blue screw with the right arm`（episode 600，起始帧 0）
- `place the blue screw at the center transfer position with the right arm`（episode 600，起始帧 42）
- `pick up the blue screw from the center transfer position with the left arm`（episode 600，起始帧 88）
- `tighten the blue screw with the left arm`（episode 600，起始帧 135）
- `pick up the white screw with the left arm`（episode 600，起始帧 512）
- `place the white screw at the center transfer position with the left arm`（episode 600，起始帧 545）
- `pick up the white screw from the center transfer position with the right arm`（episode 600，起始帧 590）
- `tighten the white screw with the right arm`（episode 600，起始帧 640）
- `pick up the red screw with the left arm`（episode 600，起始帧 1009）
- `place the red screw at the center transfer position with the left arm`（episode 600，起始帧 1047）
- `pick up the red screw from the center transfer position with the right arm`（episode 600，起始帧 1093）
- `tighten the red screw with the right arm`（episode 600，起始帧 1144）
- `retract the right arm`（episode 600，起始帧 1485）
- `insert the red screw into the red nut with the left arm`（episode 601，起始帧 38）
- `tighten the red screw with the left arm`（episode 601，起始帧 88）
- `pick up the yellow screw with the right arm`（episode 601，起始帧 409）
- `insert the yellow screw into the yellow nut with the right arm`（episode 601，起始帧 447）
- `tighten the yellow screw with the right arm`（episode 601，起始帧 484）
- `retract the arms`（episode 601，起始帧 1295）
- `pick up the white screw with the right arm`（episode 602，起始帧 404）
- `place the white screw at the center transfer position with the right arm`（episode 602，起始帧 449）
- `pick up the white screw from the center transfer position with the left arm`（episode 602，起始帧 488）
- `insert the white screw into the white nut with the left arm`（episode 602，起始帧 540）
- `tighten the white screw with the left arm`（episode 602，起始帧 583）
- `pick up the purple screw with the right arm`（episode 602，起始帧 908）
- `insert the purple screw into the purple nut with the right arm`（episode 602，起始帧 952）
- `tighten the purple screw with the right arm`（episode 602，起始帧 990）
- `pick up the red screw with the right arm`（episode 603，起始帧 0）
- `place the red screw at the center transfer position with the right arm`（episode 603，起始帧 42）
- `pick up the red screw from the center transfer position with the left arm`（episode 603，起始帧 91）
- `place the purple screw at the center transfer position with the right arm`（episode 603，起始帧 554）
- `pick up the purple screw from the center transfer position with the left arm`（episode 603，起始帧 599）
- `insert the purple screw into the purple nut with the left arm`（episode 603，起始帧 646）
- `tighten the purple screw with the left arm`（episode 603，起始帧 694）
- `place the yellow screw at the center transfer position with the right arm`（episode 604，起始帧 42）
- `pick up the yellow screw from the center transfer position with the left arm`（episode 604，起始帧 85）
- `insert the yellow screw into the yellow nut with the left arm`（episode 604，起始帧 134）
- `tighten the yellow screw with the left arm`（episode 604，起始帧 181）
- `insert the red screw into the red nut with the right arm`（episode 604，起始帧 1051）
- `pick up the yellow screw with the left arm`（episode 605，起始帧 505）
- `place the yellow screw at the center transfer position with the left arm`（episode 605，起始帧 549）
- `pick up the yellow screw from the center transfer position with the right arm`（episode 605，起始帧 591）
- `insert the blue screw into the blue nut with the right arm`（episode 605，起始帧 1040）
- `tighten the blue screw with the right arm`（episode 605，起始帧 1069）
- `insert the white screw into the white nut with the right arm`（episode 606，起始帧 455）
- `pick up the purple screw with the left arm`（episode 606，起始帧 825）
- `place the purple screw at the center transfer position with the left arm`（episode 606，起始帧 872）
- `pick up the purple screw from the center transfer position with the right arm`（episode 606，起始帧 911）
- `pick up the blue screw with the left arm`（episode 607，起始帧 923）
- `place the blue screw at the center transfer position with the left arm`（episode 607，起始帧 970）
- `pick up the blue screw from the center transfer position with the right arm`（episode 607，起始帧 1012）
- `insert the blue screw into the blue nut with the left arm`（episode 609，起始帧 42）

### Task 07（20 条）

L3：Place the four eggs from the basket into the egg holder, then close the lid.

- `pick up the first egg from the basket with the right arm`（episode 700，起始帧 0）
- `place the first egg into the egg holder with the right arm`（episode 700，起始帧 46）
- `pick up the second egg from the basket with the right arm`（episode 700，起始帧 122）
- `place the second egg into the egg holder with the right arm`（episode 700，起始帧 174）
- `pick up the third egg from the basket with the right arm`（episode 700，起始帧 230）
- `place the third egg into the egg holder with the right arm`（episode 700，起始帧 278）
- `pick up the fourth egg from the basket with the right arm`（episode 700，起始帧 333）
- `place the fourth egg into the egg holder with the right arm`（episode 700，起始帧 372）
- `close the egg holder lid with the right arm`（episode 700，起始帧 441）
- `retract the right arm`（episode 700，起始帧 477）
- `pick up the first egg from the basket with the left arm`（episode 701，起始帧 0）
- `place the first egg into the egg holder with the left arm`（episode 701，起始帧 41）
- `pick up the second egg from the basket with the left arm`（episode 701，起始帧 94）
- `place the second egg into the egg holder with the left arm`（episode 701，起始帧 141）
- `pick up the third egg from the basket with the left arm`（episode 701，起始帧 190）
- `place the third egg into the egg holder with the left arm`（episode 701，起始帧 245）
- `pick up the fourth egg from the basket with the left arm`（episode 701，起始帧 292）
- `place the fourth egg into the egg holder with the left arm`（episode 701，起始帧 339）
- `close the egg holder lid with the left arm`（episode 701，起始帧 390）
- `retract the left arm`（episode 701，起始帧 419）

### Task 08（6 条）

L3：Hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.

- `pick up the pen holder`（episode 800，起始帧 0）
- `pick up a pen`（episode 800，起始帧 91）
- `place the pen in the pen holder`（episode 800，起始帧 192）
- `place the pen holder on the table`（episode 800，起始帧 829）
- `retract the left arm`（episode 800，起始帧 884）
- `retract the right arm`（episode 807，起始帧 736）

### Task 09（7 条）

L3：Fold the clothes neatly.

- `pick up the left side of the garment`（episode 900，起始帧 0）
- `fold the left side of the garment toward the center`（episode 900，起始帧 45）
- `pick up the right side of the garment`（episode 900，起始帧 87）
- `fold the right side of the garment toward the center`（episode 900，起始帧 147）
- `pick up the bottom of the garment with both hands`（episode 900，起始帧 189）
- `fold the bottom of the garment upward`（episode 900，起始帧 233）
- `retract both arms`（episode 900，起始帧 288）

### Task 10（110 条）

L3：Hang all the mugs on the mug rack.

- `pick up the yellow mug`（episode 1000，起始帧 0）
- `hang the yellow mug on the mug rack`（episode 1000，起始帧 84）
- `pick up the white mug`（episode 1000，起始帧 180）
- `hang the white mug on the mug rack`（episode 1000，起始帧 248）
- `pick up the purple mug`（episode 1000，起始帧 341）
- `hang the purple mug on the mug rack`（episode 1000，起始帧 401）
- `retract the left arm`（episode 1000，起始帧 500）
- `pick up the brown mug`（episode 1001，起始帧 0）
- `hang the brown mug on the mug rack`（episode 1001，起始帧 72）
- `pick up the left white mug`（episode 1002，起始帧 0）
- `hang the left white mug on the mug rack`（episode 1002，起始帧 109）
- `pick up the patterned mug`（episode 1002，起始帧 288）
- `hang the patterned mug on the mug rack`（episode 1002，起始帧 387）
- `pick up the right white mug`（episode 1002，起始帧 532）
- `hang the right white mug on the mug rack`（episode 1002，起始帧 651）
- `retract the right arm`（episode 1002，起始帧 802）
- `pick up the green mug`（episode 1003，起始帧 351）
- `hang the green mug on the mug rack`（episode 1003，起始帧 461）
- `pick up the beige mug`（episode 1004，起始帧 0）
- `hang the beige mug on the mug rack`（episode 1004，起始帧 88）
- `pick up the middle brown mug`（episode 1004，起始帧 179）
- `hang the middle brown mug on the mug rack`（episode 1004，起始帧 250）
- `pick up the left brown mug`（episode 1004，起始帧 398）
- `hang the left brown mug on the mug rack`（episode 1004，起始帧 486）
- `pick up the black mug`（episode 1005，起始帧 138）
- `hang the black mug on the mug rack`（episode 1005，起始帧 207）
- `pick up the right yellow mug`（episode 1006，起始帧 0）
- `hang the right yellow mug on the mug rack`（episode 1006，起始帧 74）
- `pick up the center yellow mug`（episode 1006，起始帧 184）
- `hang the center yellow mug on the mug rack`（episode 1006，起始帧 255）
- `pick up the white mug with red handle`（episode 1013，起始帧 0）
- `hang the white mug with red handle on the mug rack`（episode 1013，起始帧 87）
- `pick up the white mug with burger print`（episode 1013，起始帧 588）
- `hang the white mug with burger print on the mug rack`（episode 1013，起始帧 733）
- `pick up the white mug with hamburger print`（episode 1015，起始帧 296）
- `hang the white mug with hamburger print on the mug rack`（episode 1015，起始帧 353）
- `pick up the dark brown mug`（episode 1021，起始帧 0）
- `hang the dark brown mug on the mug rack`（episode 1021，起始帧 44）
- `pick up the middle white mug`（episode 1024，起始帧 180）
- `hang the middle white mug on the mug rack`（episode 1024，起始帧 249）
- `pick up the grey mug`（episode 1028，起始帧 313）
- `hang the grey mug on the mug rack`（episode 1028，起始帧 368）
- `pick up the middle green mug`（episode 1029，起始帧 169）
- `hang the middle green mug on the mug rack`（episode 1029，起始帧 220）
- `pick up the left green mug`（episode 1029，起始帧 317）
- `hang the left green mug on the mug rack`（episode 1029，起始帧 380）
- `pick up the left mug`（episode 1030，起始帧 0）
- `hang the left mug on the mug rack`（episode 1030，起始帧 67）
- `pick up the upper mug`（episode 1030，起始帧 132）
- `hang the upper mug on the mug rack`（episode 1030，起始帧 251）
- `pick up the lower mug`（episode 1030，起始帧 340）
- `place the lower mug in front of the mug rack`（episode 1030，起始帧 411）
- `pick up the lower mug from in front of the mug rack`（episode 1030，起始帧 464）
- `hang the lower mug on the mug rack`（episode 1030，起始帧 538）
- `place the white mug in front of the mug rack`（episode 1031，起始帧 486）
- `pick up the white mug from in front of the mug rack`（episode 1031，起始帧 542）
- `pick up the big white mug`（episode 1033，起始帧 0）
- `hang the big white mug on the mug rack`（episode 1033，起始帧 95）
- `pick up the red-handled mug`（episode 1035，起始帧 178）
- `hang the red-handled mug on the mug rack`（episode 1035，起始帧 271）
- `pick up the polka dot mug`（episode 1037，起始帧 196）
- `hang the polka dot mug on the mug rack`（episode 1037，起始帧 257）
- `pick up the right grey mug`（episode 1042，起始帧 0）
- `hang the right grey mug on the mug rack`（episode 1042，起始帧 97）
- `pick up the left grey mug`（episode 1042，起始帧 319）
- `hang the left grey mug on the mug rack`（episode 1042，起始帧 394）
- `pick up the red mug`（episode 1043，起始帧 173）
- `hang the red mug on the mug rack`（episode 1043，起始帧 222）
- `pick up the spotted mug`（episode 1043，起始帧 313）
- `hang the spotted mug on the mug rack`（episode 1043，起始帧 377）
- `pick up the pink mug`（episode 1044，起始帧 0）
- `hang the pink mug on the mug rack`（episode 1044，起始帧 59）
- `place the brown mug in front of the mug rack`（episode 1044，起始帧 268）
- `pick up the brown mug from in front of the mug rack`（episode 1044，起始帧 310）
- `pick up the dotted mug`（episode 1045，起始帧 131）
- `hang the dotted mug on the mug rack`（episode 1045，起始帧 198）
- `pick up the upper white mug`（episode 1047，起始帧 0）
- `hang the upper white mug on the mug rack`（episode 1047，起始帧 101）
- `pick up the lower white mug`（episode 1047，起始帧 180）
- `hang the lower white mug on the mug rack`（episode 1047，起始帧 241）
- `pick up the right black mug`（episode 1052，起始帧 0）
- `hang the right black mug on the mug rack`（episode 1052，起始帧 88）
- `pick up the left black mug`（episode 1052，起始帧 185）
- `hang the left black mug on the mug rack`（episode 1052，起始帧 256）
- `pick up the right floral mug`（episode 1053，起始帧 0）
- `hang the right floral mug on the mug rack`（episode 1053，起始帧 87）
- `pick up the left floral mug`（episode 1053，起始帧 300）
- `hang the left floral mug on the mug rack`（episode 1053，起始帧 407）
- `pick up the cream mug`（episode 1054，起始帧 363）
- `hang the cream mug on the mug rack`（episode 1054，起始帧 443）
- `pick up the red patterned mug`（episode 1058，起始帧 0）
- `hang the red patterned mug on the mug rack`（episode 1058，起始帧 94）
- `place the yellow mug in front of the mug rack`（episode 1065，起始帧 386）
- `pick up the yellow mug from in front of the mug rack`（episode 1065，起始帧 430）
- `pick up the left yellow mug`（episode 1068，起始帧 309）
- `hang the left yellow mug on the mug rack`（episode 1068，起始帧 385）
- `pick up the large white mug`（episode 1070，起始帧 0）
- `hang the large white mug on the mug rack`（episode 1070，起始帧 82）
- `pick up the small white mug`（episode 1070，起始帧 296）
- `hang the small white mug on the mug rack`（episode 1070，起始帧 370）
- `pick up the floral mug`（episode 1078，起始帧 297）
- `hang the floral mug on the mug rack`（episode 1078，起始帧 384）
- `pick up the right mug`（episode 1088，起始帧 0）
- `hang the right mug on the mug rack`（episode 1088，起始帧 90）
- `pick up the center mug`（episode 1088，起始帧 383）
- `hang the center mug on the mug rack`（episode 1088，起始帧 433）
- `pick up the middle mug`（episode 1092，起始帧 176）
- `hang the middle mug on the mug rack`（episode 1092，起始帧 239）
- `pick up the red-handle mug`（episode 1097，起始帧 328）
- `hang the red-handle mug on the mug rack`（episode 1097，起始帧 359）

### Task 11（37 条）

L3：Observe the object placement order, remember it, then place the corresponding objects into the basket in the same order.

- `wait for the opponent to place the next object`（episode 1100，起始帧 0）
- `opponent places doll`（episode 1100，起始帧 41）
- `opponent places camera`（episode 1100，起始帧 154）
- `opponent places watch`（episode 1100，起始帧 234）
- `opponent places car`（episode 1100，起始帧 329）
- `opponent places phone`（episode 1100，起始帧 410）
- `pick up doll from table with the right arm`（episode 1100，起始帧 453）
- `place doll in the center with the right arm`（episode 1100，起始帧 510）
- `pick up doll from center with the left arm`（episode 1100，起始帧 553）
- `place doll in the left basket with the left arm`（episode 1100，起始帧 596）
- `pick up camera from table with the left arm`（episode 1100，起始帧 643）
- `place camera in the left basket with the left arm`（episode 1100，起始帧 680）
- `pick up watch with the right arm`（episode 1100，起始帧 720）
- `place watch in the center with the right arm`（episode 1100，起始帧 763）
- `pick up watch from center with the left arm`（episode 1100，起始帧 808）
- `place watch in the left basket with the left arm`（episode 1100，起始帧 857）
- `pick up car from table with the left arm`（episode 1100，起始帧 903）
- `place car in the left basket with the left arm`（episode 1100，起始帧 938）
- `pick up phone with the right arm`（episode 1100，起始帧 979）
- `place phone in the center with the right arm`（episode 1100，起始帧 1023）
- `pick up phone from center with the left arm`（episode 1100，起始帧 1061）
- `place phone in the left basket with the left arm`（episode 1100，起始帧 1106）
- `retract both arms`（episode 1100，起始帧 1156）
- `pick up watch from table with the left arm`（episode 1101，起始帧 418）
- `pick up camera with the right arm`（episode 1101，起始帧 515）
- `place camera in the center with the right arm`（episode 1101，起始帧 559）
- `pick up camera from center with the left arm`（episode 1101，起始帧 602）
- `pick up doll with the right arm`（episode 1101，起始帧 956）
- `pick up phone from table with the left arm`（episode 1102，起始帧 475）
- `pick up camera from table with the right arm`（episode 1103，起始帧 462）
- `pick up car with the right arm`（episode 1103，起始帧 1140）
- `place car in the center with the right arm`（episode 1103，起始帧 1185）
- `pick up car from center with the left arm`（episode 1103，起始帧 1229）
- `pick up car from table with the right arm`（episode 1105，起始帧 422）
- `pick up doll from table with the left arm`（episode 1105，起始帧 1169）
- `pick up watch from table with the right arm`（episode 1108，起始帧 407）
- `pick up phone from table with the right arm`（episode 1112，起始帧 448）

### Task 12（8 条）

L3：Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.

- `pick up the key with the right hand`（episode 1200，起始帧 0）
- `handover the key from the right hand to the left hand`（episode 1200，起始帧 33）
- `insert the key into the keyhole with the left hand`（episode 1200，起始帧 115）
- `turn the key with the left hand`（episode 1200，起始帧 163）
- `pick up the key with the left hand`（episode 1207，起始帧 0）
- `handover the key from the left hand to the right hand`（episode 1207，起始帧 32）
- `insert the key into the keyhole with the right hand`（episode 1207，起始帧 116）
- `turn the key with the right hand`（episode 1207，起始帧 165）

### Task 13（7 条）

L3：Insert the three tubes into the rack one by one.

- `pick up the middle tube`（episode 1300，起始帧 0）
- `insert the middle tube into the rack`（episode 1300，起始帧 38）
- `pick up the right tube`（episode 1300，起始帧 89）
- `insert the right tube into the rack`（episode 1300，起始帧 137）
- `pick up the left tube`（episode 1300，起始帧 190）
- `insert the left tube into the rack`（episode 1300，起始帧 237）
- `retract the left arm`（episode 1300，起始帧 289）

### Task 14（8 条）

L3：Wait for the opponent to discard a tile, then declare a kong with the matching tiles.

- `wait for the opponent to discard a tile`（episode 1400，起始帧 0）
- `push the first tile matching the opponent's discarded tile forward`（episode 1400，起始帧 60）
- `push the second tile matching the opponent's discarded tile forward`（episode 1400，起始帧 114）
- `push the third tile matching the opponent's discarded tile forward`（episode 1400，起始帧 155）
- `pick up a replacement tile from the wall`（episode 1400，起始帧 212）
- `hand over the replacement tile`（episode 1400，起始帧 253）
- `place the replacement tile behind the current hand`（episode 1400，起始帧 339）
- `retract both arms`（episode 1403，起始帧 367）

### Task 15（8 条）

L3：Pick up two slices of bread, place them into the toaster, and press the lever down.

- `pick up the first bread slice`（episode 1500，起始帧 0）
- `handover the first bread slice`（episode 1500，起始帧 79）
- `place the first bread slice into the toaster`（episode 1500，起始帧 173）
- `pick up the second bread slice`（episode 1500，起始帧 280）
- `handover the second bread slice`（episode 1500，起始帧 341）
- `place the second bread slice into the toaster`（episode 1500，起始帧 439）
- `press the toaster lever down`（episode 1500，起始帧 511）
- `retract both arms`（episode 1500，起始帧 580）

### Task 16（27 条）

L3：Remember the first object on the conveyor, then pick the matching object when it appears again.

- `observe the green frog rattle, remember it, and wait for it to reappear`（episode 1600，起始帧 0）
- `reach for and pick up the green frog rattle with the right arm`（episode 1600，起始帧 202）
- `observe the blue mobile phone, remember it, and wait for it to reappear`（episode 1602，起始帧 0）
- `reach for and pick up the blue mobile phone with the right arm`（episode 1602，起始帧 282）
- `observe the blue and silver energy drink can, remember it, and wait for it to reappear`（episode 1604，起始帧 0）
- `reach for and pick up the blue and silver energy drink can with the right arm`（episode 1604，起始帧 275）
- `observe the white patterned paper plate, remember it, and wait for it to reappear`（episode 1606，起始帧 0）
- `reach for and pick up the white patterned paper plate with the right arm`（episode 1606，起始帧 176）
- `observe the white-haired anime figurine, remember it, and wait for it to reappear`（episode 1609，起始帧 0）
- `reach for and pick up the white-haired anime figurine with the right arm`（episode 1609，起始帧 280）
- `observe the black camera, remember it, and wait for it to reappear`（episode 1611，起始帧 0）
- `reach for and pick up the black camera with the right arm`（episode 1611，起始帧 183）
- `observe the white patterned paper bowl, remember it, and wait for it to reappear`（episode 1612，起始帧 0）
- `reach for and pick up the white patterned paper bowl with the right arm`（episode 1612，起始帧 214）
- `observe the white patterned paper cup, remember it, and wait for it to reappear`（episode 1623，起始帧 0）
- `reach for and pick up the white patterned paper cup with the right arm`（episode 1623，起始帧 229）
- `reach for and pick up the white-haired anime figurine with the left arm`（episode 1630，起始帧 267）
- `reach for and pick up the blue and silver energy drink can with the left arm`（episode 1631，起始帧 312）
- `reach for and pick up the black camera with the left arm`（episode 1633，起始帧 171）
- `reach for and pick up the green frog rattle with the left arm`（episode 1634，起始帧 300）
- `observe the brown handbag, remember it, and wait for it to reappear`（episode 1636，起始帧 0）
- `reach for and pick up the brown handbag with the right arm`（episode 1636，起始帧 209）
- `reach for and pick up the blue mobile phone with the left arm`（episode 1637，起始帧 202）
- `reach for and pick up the white patterned paper plate with the left arm`（episode 1644，起始帧 174）
- `reach for and pick up the white patterned paper cup with the left arm`（episode 1649，起始帧 261）
- `observe the brown headphones, remember it, and wait for it to reappear`（episode 1669，起始帧 0）
- `reach for and pick up the brown headphones with the right arm`（episode 1669，起始帧 198）

### Task 17（8 条）

L3：Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside.

- `pick up the alarm clock`（episode 1700，起始帧 0）
- `place the alarm clock on the top of drawer`（episode 1700，起始帧 74）
- `pick up the figurine`（episode 1700，起始帧 132）
- `place the figurine on the figurine stand`（episode 1700，起始帧 249）
- `pick up the mouse`（episode 1700，起始帧 357）
- `place the mouse on the mouse pad`（episode 1700，起始帧 450）
- `push the keyboard into the keyboard frame`（episode 1700，起始帧 494）
- `retract both arms`（episode 1700，起始帧 822）

### Task 18（16 条）

L3：Place all the objects into the box with their front sides facing left.

- `pick up the hammer`（episode 1800，起始帧 0）
- `place the hammer in an open area of the table`（episode 1800，起始帧 74）
- `pick up the shoe`（episode 1800，起始帧 121）
- `place the shoe in an open area of the table`（episode 1800，起始帧 178）
- `pick up the box`（episode 1800，起始帧 230）
- `place the box at the center of the table`（episode 1800，起始帧 296）
- `place the shoe in the box with its front facing left`（episode 1800，起始帧 454）
- `place the hammer in the box with its front facing left`（episode 1800，起始帧 640）
- `pick up the toothbrush`（episode 1800，起始帧 785）
- `place the toothbrush in the box with its front facing left`（episode 1800，起始帧 867）
- `pick up the toy car`（episode 1800，起始帧 948）
- `place the toy car in the box with its front facing left`（episode 1800，起始帧 1031）
- `retract the arms`（episode 1800，起始帧 1120）
- `place the toothbrush in an open area of the table`（episode 1802，起始帧 142）
- `place the toy car in an open area of the table`（episode 1803，起始帧 137）
- `place the box in the center`（episode 1841，起始帧 385）

### Task 19（10 条）

L3：Pick up the mallet and strike all xylophone keys from left to right.

- `pick up the mallet with the left hand`（episode 1900，起始帧 0）
- `strike the red key with the left hand`（episode 1900，起始帧 41）
- `strike the orange key with the left hand`（episode 1900，起始帧 82）
- `strike the yellow key with the left hand`（episode 1900，起始帧 100）
- `strike the green key with the left hand`（episode 1900，起始帧 118）
- `handover the mallet from the left hand to the right hand`（episode 1900，起始帧 137）
- `strike the light blue key with the right hand`（episode 1900，起始帧 211）
- `strike the dark blue key with the right hand`（episode 1900，起始帧 254）
- `strike the purple key with the right hand`（episode 1900，起始帧 270）
- `strike the pink key with the right hand`（episode 1900，起始帧 288）

### Task 20（21 条）

L3：Place all stacking toy pieces onto the correct pegs.

- `pick up orange round ring`（episode 2000，起始帧 0）
- `insert orange round ring onto the first peg`（episode 2000，起始帧 38）
- `pick up yellow square piece`（episode 2000，起始帧 328）
- `insert yellow square piece onto the second peg`（episode 2000，起始帧 361）
- `pick up purple star piece`（episode 2000，起始帧 588）
- `insert purple star piece onto the third peg`（episode 2000，起始帧 625）
- `pick up blue rectangular piece`（episode 2000，起始帧 752）
- `insert blue rectangular piece onto the fourth peg`（episode 2000，起始帧 808）
- `retract both arms`（episode 2000，起始帧 862）
- `insert blue rectangular piece onto the first peg`（episode 2001，起始帧 38）
- `insert orange round ring onto the second peg`（episode 2001，起始帧 382）
- `insert yellow square piece onto the third peg`（episode 2001，起始帧 642）
- `insert purple star piece onto the fourth peg`（episode 2001，起始帧 829）
- `insert yellow square piece onto the first peg`（episode 2002，起始帧 37）
- `insert blue rectangular piece onto the second peg`（episode 2003，起始帧 395）
- `insert orange round ring onto the third peg`（episode 2003，起始帧 659）
- `insert purple star piece onto the first peg`（episode 2005，起始帧 41）
- `insert orange round ring onto the fourth peg`（episode 2005，起始帧 852）
- `insert purple star piece onto the second peg`（episode 2008，起始帧 372）
- `insert blue rectangular piece onto the third peg`（episode 2010，起始帧 671）
- `insert yellow square piece onto the fourth peg`（episode 2012，起始帧 822）

### Task 21（12 条）

L3：Play tic-tac-toe as the first player and fill the board with the opponent.

- `pick up an O piece`（episode 2100，起始帧 0）
- `place the O piece in the top center cell`（episode 2100，起始帧 30）
- `wait for the opponent to place a piece until the numbers of O and X pieces on the grid are equal`（episode 2100，起始帧 107）
- `place the O piece in the bottom center cell`（episode 2100，起始帧 227）
- `place the O piece in the top right cell`（episode 2100，起始帧 409）
- `place the O piece in the bottom left cell`（episode 2100，起始帧 618）
- `place the O piece in the center cell`（episode 2100，起始帧 810）
- `retract both arms`（episode 2100，起始帧 854）
- `place the O piece in the bottom right cell`（episode 2101，起始帧 30）
- `place the O piece in the middle right cell`（episode 2101，起始帧 828）
- `place the O piece in the middle left cell`（episode 2103，起始帧 30）
- `place the O piece in the top left cell`（episode 2103，起始帧 233）

### Task 22（7 条）

L3：Plug the charger into the power strip.

- `pick up the charger with the right arm`（episode 2200，起始帧 0）
- `handover the charger from the right arm to the left arm`（episode 2200，起始帧 33）
- `insert the charger into the power strip`（episode 2200，起始帧 122）
- `retract the left arm`（episode 2200，起始帧 174）
- `pick up the charger with the left arm`（episode 2201，起始帧 0）
- `handover the charger from the left arm to the right arm`（episode 2201，起始帧 33）
- `retract the right arm`（episode 2201，起始帧 170）

### Task 23（8 条）

L3：Pour all the balls from the cup into the vase.

- `pick up the cup with the right arm`（episode 2300，起始帧 0）
- `handover the cup from the right arm to the left arm`（episode 2300，起始帧 69）
- `move the cup to the vase and pour all balls into it`（episode 2300，起始帧 195）
- `return the empty cup and place it on the table`（episode 2300，起始帧 264）
- `retract the left arm`（episode 2300，起始帧 320）
- `pick up the cup with the left arm`（episode 2301，起始帧 0）
- `handover the cup from the left arm to the right arm`（episode 2301，起始帧 82）
- `retract the right arm`（episode 2301，起始帧 337）

### Task 24（4 条）

L3：Pour the liquid from the bottle into the cup.

- `pick up the bottle with the left arm`（episode 2400，起始帧 0）
- `move the bottle above the cup and pour the liquid into it`（episode 2400，起始帧 37）
- `return the bottle toward its starting position`（episode 2400，起始帧 124）
- `pick up the bottle with the right arm`（episode 2411，起始帧 0）

### Task 25（93 条）

L3：Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm.

- `press the left red button (1 of 9)`（episode 2500，起始帧 0）
- `press the left red button (2 of 9)`（episode 2500，起始帧 55）
- `press the left red button (3 of 9)`（episode 2500，起始帧 69）
- `press the left red button (4 of 9)`（episode 2500，起始帧 84）
- `press the left red button (5 of 9)`（episode 2500，起始帧 99）
- `press the left red button (6 of 9)`（episode 2500，起始帧 113）
- `press the left red button (7 of 9)`（episode 2500，起始帧 130）
- `press the left red button (8 of 9)`（episode 2500，起始帧 144）
- `press the left red button (9 of 9)`（episode 2500，起始帧 159）
- `press the blue confirmation button for the left red button`（episode 2500，起始帧 184）
- `press the right red button (1 of 1)`（episode 2500，起始帧 241）
- `press the blue confirmation button for the right red button`（episode 2500，起始帧 298）
- `retract the right arm`（episode 2500，起始帧 357）
- `press the left red button (1 of 6)`（episode 2501，起始帧 0）
- `press the left red button (2 of 6)`（episode 2501，起始帧 55）
- `press the left red button (3 of 6)`（episode 2501，起始帧 70）
- `press the left red button (4 of 6)`（episode 2501，起始帧 84）
- `press the left red button (5 of 6)`（episode 2501，起始帧 101）
- `press the left red button (6 of 6)`（episode 2501，起始帧 115）
- `press the right red button (1 of 5)`（episode 2501，起始帧 196）
- `press the right red button (2 of 5)`（episode 2501，起始帧 243）
- `press the right red button (3 of 5)`（episode 2501，起始帧 257）
- `press the right red button (4 of 5)`（episode 2501，起始帧 272）
- `press the right red button (5 of 5)`（episode 2501，起始帧 287）
- `press the left red button (1 of 4)`（episode 2502，起始帧 0）
- `press the left red button (2 of 4)`（episode 2502，起始帧 55）
- `press the left red button (3 of 4)`（episode 2502，起始帧 70）
- `press the left red button (4 of 4)`（episode 2502，起始帧 84）
- `press the right red button (1 of 9)`（episode 2502，起始帧 166）
- `press the right red button (2 of 9)`（episode 2502，起始帧 212）
- `press the right red button (3 of 9)`（episode 2502，起始帧 227）
- `press the right red button (4 of 9)`（episode 2502，起始帧 241）
- `press the right red button (5 of 9)`（episode 2502，起始帧 256）
- `press the right red button (6 of 9)`（episode 2502，起始帧 272）
- `press the right red button (7 of 9)`（episode 2502，起始帧 288）
- `press the right red button (8 of 9)`（episode 2502，起始帧 303）
- `press the right red button (9 of 9)`（episode 2502，起始帧 320）
- `press the left red button (1 of 1)`（episode 2504，起始帧 0）
- `press the left red button (1 of 7)`（episode 2505，起始帧 0）
- `press the left red button (2 of 7)`（episode 2505，起始帧 55）
- `press the left red button (3 of 7)`（episode 2505，起始帧 70）
- `press the left red button (4 of 7)`（episode 2505，起始帧 84）
- `press the left red button (5 of 7)`（episode 2505，起始帧 100）
- `press the left red button (6 of 7)`（episode 2505，起始帧 115）
- `press the left red button (7 of 7)`（episode 2505，起始帧 129）
- `press the right red button (1 of 7)`（episode 2505，起始帧 213）
- `press the right red button (2 of 7)`（episode 2505，起始帧 259）
- `press the right red button (3 of 7)`（episode 2505，起始帧 274）
- `press the right red button (4 of 7)`（episode 2505，起始帧 288）
- `press the right red button (5 of 7)`（episode 2505，起始帧 304）
- `press the right red button (6 of 7)`（episode 2505，起始帧 319）
- `press the right red button (7 of 7)`（episode 2505，起始帧 335）
- `press the left red button (1 of 2)`（episode 2506，起始帧 0）
- `press the left red button (2 of 2)`（episode 2506，起始帧 55）
- `press the left red button (1 of 3)`（episode 2509，起始帧 0）
- `press the left red button (2 of 3)`（episode 2509，起始帧 55）
- `press the left red button (3 of 3)`（episode 2509，起始帧 70）
- `press the right red button (1 of 2)`（episode 2510，起始帧 141）
- `press the right red button (2 of 2)`（episode 2510，起始帧 187）
- `press the left red button (1 of 8)`（episode 2511，起始帧 0）
- `press the left red button (2 of 8)`（episode 2511，起始帧 55）
- `press the left red button (3 of 8)`（episode 2511，起始帧 71）
- `press the left red button (4 of 8)`（episode 2511，起始帧 87）
- `press the left red button (5 of 8)`（episode 2511，起始帧 102）
- `press the left red button (6 of 8)`（episode 2511，起始帧 121）
- `press the left red button (7 of 8)`（episode 2511，起始帧 136）
- `press the left red button (8 of 8)`（episode 2511，起始帧 152）
- `press the right red button (1 of 8)`（episode 2511，起始帧 238）
- `press the right red button (2 of 8)`（episode 2511，起始帧 284）
- `press the right red button (3 of 8)`（episode 2511，起始帧 299）
- `press the right red button (4 of 8)`（episode 2511，起始帧 313）
- `press the right red button (5 of 8)`（episode 2511，起始帧 328）
- `press the right red button (6 of 8)`（episode 2511，起始帧 343）
- `press the right red button (7 of 8)`（episode 2511，起始帧 359）
- `press the right red button (8 of 8)`（episode 2511，起始帧 376）
- `press the left red button (1 of 5)`（episode 2512，起始帧 0）
- `press the left red button (2 of 5)`（episode 2512，起始帧 55）
- `press the left red button (3 of 5)`（episode 2512，起始帧 70）
- `press the left red button (4 of 5)`（episode 2512，起始帧 84）
- `press the left red button (5 of 5)`（episode 2512，起始帧 100）
- `press the right red button (1 of 6)`（episode 2513，起始帧 152）
- `press the right red button (2 of 6)`（episode 2513，起始帧 198）
- `press the right red button (3 of 6)`（episode 2513，起始帧 213）
- `press the right red button (4 of 6)`（episode 2513，起始帧 228）
- `press the right red button (5 of 6)`（episode 2513，起始帧 242）
- `press the right red button (6 of 6)`（episode 2513，起始帧 257）
- `press the right red button (1 of 4)`（episode 2518，起始帧 155）
- `press the right red button (2 of 4)`（episode 2518，起始帧 202）
- `press the right red button (3 of 4)`（episode 2518，起始帧 217）
- `press the right red button (4 of 4)`（episode 2518，起始帧 232）
- `press the right red button (1 of 3)`（episode 2519，起始帧 122）
- `press the right red button (2 of 3)`（episode 2519，起始帧 168）
- `press the right red button (3 of 3)`（episode 2519，起始帧 182）

### Task 26（7 条）

L3：Push the T-shaped block to align it precisely with the gray T-shaped pad.

- `push the T-shaped block onto the gray T-shaped pad with the right arm`（episode 2600，起始帧 0）
- `retract the right arm`（episode 2600，起始帧 349）
- `push the T-shaped block close to the gray T-shaped pad with the left arm`（episode 2601，起始帧 0）
- `retract both arms`（episode 2601，起始帧 504）
- `push the T-shaped block close to the gray T-shaped pad with the right arm`（episode 2602，起始帧 0）
- `push the T-shaped block onto the gray T-shaped pad with the left arm`（episode 2602，起始帧 126）
- `retract the left arm`（episode 2607，起始帧 225）

### Task 27（6 条）

L3：Pick up the bottles and throw them into the dustbin, using handover when needed.

- `pick up a bottle with the left arm`（episode 2700，起始帧 0）
- `throw the bottle into the dustbin with the left arm`（episode 2700，起始帧 22）
- `retract both arms`（episode 2700，起始帧 344）
- `pick up a bottle with the right arm`（episode 2701，起始帧 124）
- `handover the bottle from the right arm to the left arm`（episode 2701，起始帧 309）
- `throw the bottle into the dustbin with the right arm`（episode 2738，起始帧 296）

### Task 28（23 条）

L3：Arrange the five nesting dolls in a row from left to right, from smallest to largest.

- `pick up the smallest doll with the right arm`（episode 2800，起始帧 0）
- `place the smallest doll in the center with the right arm`（episode 2800，起始帧 32）
- `pick up the smallest doll from the center with the left arm`（episode 2800，起始帧 83）
- `place the smallest doll in the first position with the left arm`（episode 2800，起始帧 131）
- `pick up the second smallest doll with the right arm`（episode 2800，起始帧 178）
- `place the second smallest doll in the center with the right arm`（episode 2800，起始帧 224）
- `pick up the second smallest doll from the center with the left arm`（episode 2800，起始帧 270）
- `place the second smallest doll in the second position with the left arm`（episode 2800，起始帧 318）
- `pick up the middle doll with the right arm`（episode 2800，起始帧 360）
- `place the middle doll in the third position with the right arm`（episode 2800，起始帧 405）
- `pick up the second largest doll with the left arm`（episode 2800，起始帧 455）
- `place the second largest doll in the center with the left arm`（episode 2800，起始帧 504）
- `pick up the second largest doll from the center with the right arm`（episode 2800，起始帧 548）
- `place the second largest doll in the fourth position with the right arm`（episode 2800，起始帧 593）
- `pick up the largest doll with the right arm`（episode 2800，起始帧 635）
- `place the largest doll in the fifth position with the right arm`（episode 2800，起始帧 676）
- `retract right arm`（episode 2800，起始帧 719）
- `pick up the middle doll with the left arm`（episode 2801，起始帧 361）
- `place the middle doll in the third position with the left arm`（episode 2801，起始帧 397）
- `pick up the smallest doll with the left arm`（episode 2802，起始帧 0）
- `pick up the second smallest doll with the left arm`（episode 2802，起始帧 76）
- `pick up the second largest doll with the right arm`（episode 2802，起始帧 256）
- `place nesting doll 1 on the far left with the right arm`（episode 2849，起始帧 726）

### Task 29（56 条）

L3：Stack the three blocks with different textures.

- `pick up purple block`（episode 2900，起始帧 0）
- `place purple block as the bottom block`（episode 2900，起始帧 35）
- `pick up light wooden block`（episode 2900，起始帧 73）
- `stack light wooden block on purple block`（episode 2900，起始帧 119）
- `pick up striped wooden block`（episode 2900，起始帧 157）
- `stack striped wooden block on light wooden block`（episode 2900，起始帧 196）
- `retract left arm`（episode 2900，起始帧 236）
- `pick up white block`（episode 2901，起始帧 0）
- `place white block as the bottom block`（episode 2901，起始帧 37）
- `pick up black block`（episode 2901，起始帧 74）
- `stack black block on white block`（episode 2901，起始帧 120）
- `pick up wooden block`（episode 2901，起始帧 154）
- `stack wooden block on black block`（episode 2901，起始帧 200）
- `retract right arm`（episode 2901，起始帧 237）
- `place wooden block as the bottom block`（episode 2902，起始帧 42）
- `stack white block on wooden block`（episode 2902，起始帧 121）
- `stack purple block on white block`（episode 2902，起始帧 196）
- `place black block as the bottom block`（episode 2903，起始帧 36）
- `stack white block on black block`（episode 2903，起始帧 121）
- `stack wooden block on white block`（episode 2903，起始帧 191）
- `stack purple block on wooden block`（episode 2904，起始帧 199）
- `place striped wooden block as the bottom block`（episode 2910，起始帧 34）
- `stack black block on striped wooden block`（episode 2910，起始帧 109）
- `stack purple block on black block`（episode 2910，起始帧 194）
- `stack wooden block on purple block`（episode 2911，起始帧 116）
- `pick up yellow block`（episode 2912，起始帧 0）
- `place yellow block as the bottom block`（episode 2912，起始帧 37）
- `stack wooden block on yellow block`（episode 2912，起始帧 121）
- `stack black block on wooden block`（episode 2914，起始帧 116）
- `stack white block on purple block`（episode 2915，起始帧 111）
- `pick up gold block`（episode 2919，起始帧 0）
- `place gold block as the bottom block`（episode 2919，起始帧 33）
- `stack black block on gold block`（episode 2919，起始帧 117）
- `stack white block on striped wooden block`（episode 2920，起始帧 123）
- `stack striped wooden block on purple block`（episode 2922，起始帧 195）
- `stack yellow block on striped wooden block`（episode 2925，起始帧 116）
- `stack white block on yellow block`（episode 2925，起始帧 201）
- `stack black block on purple block`（episode 2926，起始帧 189）
- `place light wooden block as the bottom block`（episode 2929，起始帧 37）
- `stack purple block on light wooden block`（episode 2929，起始帧 124）
- `stack yellow block on purple block`（episode 2938，起始帧 120）
- `stack yellow block on wooden block`（episode 2940，起始帧 118）
- `stack purple block on yellow block`（episode 2943，起始帧 191）
- `stack striped wooden block on white block`（episode 2947，起始帧 115）
- `stack wooden block on striped wooden block`（episode 2947，起始帧 193）
- `pick up red block`（episode 2950，起始帧 72）
- `stack red block on purple block`（episode 2950，起始帧 117）
- `stack white block on red block`（episode 2950，起始帧 203）
- `stack purple block on striped wooden block`（episode 2955，起始帧 106）
- `stack striped wooden block on wooden block`（episode 2958，起始帧 193）
- `stack striped wooden block on yellow block`（episode 2965，起始帧 120）
- `stack striped wooden block on gold block`（episode 2967，起始帧 111）
- `stack light wooden block on white block`（episode 2975，起始帧 117）
- `stack light wooden block on black block`（episode 2977，起始帧 185）
- `stack yellow block on black block`（episode 2988，起始帧 200）
- `stack wooden block on gold block`（episode 2996，起始帧 109）

### Task 30（22 条）

L3：Stack the three bowls together.

- `pick up right bowl`（episode 3000，起始帧 0）
- `place right bowl on the middle bowl`（episode 3000，起始帧 60）
- `pick up left bowl`（episode 3000，起始帧 108）
- `place left bowl on the bowl stack`（episode 3000，起始帧 164）
- `retract left arm`（episode 3000，起始帧 226）
- `place right bowl at an unknown location`（episode 3001，起始帧 61）
- `place left bowl on the right bowl`（episode 3001，起始帧 174）
- `retry until successfully picking the middle bowl with the left arm`（episode 3001，起始帧 234）
- `place the middle bowl on the left bowl with the left arm`（episode 3001，起始帧 302）
- `pick up middle bowl`（episode 3002，起始帧 212）
- `place middle bowl on the bowl stack`（episode 3002，起始帧 271）
- `retract right arm`（episode 3002，起始帧 314）
- `place middle bowl on the right bowl`（episode 3004，起始帧 127）
- `place middle bowl at an unknown location`（episode 3006，起始帧 84）
- `place left bowl on the middle bowl`（episode 3006，起始帧 203）
- `place right bowl on the left bowl`（episode 3006，起始帧 366）
- `place right bowl on the table`（episode 3007，起始帧 98）
- `place right bowl on the bowl stack`（episode 3008，起始帧 219）
- `place middle bowl on the left bowl`（episode 3013，起始帧 52）
- `place middle bowl on the table`（episode 3017，起始帧 32）
- `place left bowl at an unknown location`（episode 3021，起始帧 51）
- `place left bowl on the table`（episode 3024，起始帧 75）

### Task 31（57 条）

L3：Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.

- `pick up the headphones`（episode 3100，起始帧 0）
- `hang the headphones on the headphone stand`（episode 3100，起始帧 42）
- `close laptop lid`（episode 3100，起始帧 98）
- `pick up laptop`（episode 3100，起始帧 215）
- `handover laptop from left to right arm`（episode 3100，起始帧 308）
- `place laptop in vertical stand`（episode 3100，起始帧 428）
- `retract right arm from vertical stand`（episode 3100，起始帧 524）
- `close laptop`（episode 3101，起始帧 95）
- `handover laptop`（episode 3101，起始帧 278）
- `place laptop`（episode 3101，起始帧 384）
- `retract right arm`（episode 3101，起始帧 470）
- `close the laptop`（episode 3102，起始帧 118）
- `pick up the laptop`（episode 3102，起始帧 269）
- `handover the laptop from left to right arm`（episode 3102，起始帧 350）
- `place the laptop in the vertical stand`（episode 3102，起始帧 508）
- `retract the right arm`（episode 3102，起始帧 635）
- `handover laptop from left to right gripper`（episode 3103，起始帧 296）
- `place laptop in the vertical stand`（episode 3103，起始帧 381）
- `handover laptop to right arm`（episode 3105，起始帧 255）
- `retract arm`（episode 3106，起始帧 542）
- `close the laptop lid`（episode 3108，起始帧 78）
- `place the laptop into vertical stand`（episode 3108，起始帧 388）
- `place laptop into vertical stand`（episode 3114，起始帧 383）
- `place laptop in stand`（episode 3117，起始帧 373）
- `close laptop screen`（episode 3118，起始帧 60）
- `pick up closed laptop`（episode 3118，起始帧 179）
- `close laptop lid with right arm`（episode 3123，起始帧 121）
- `pick up laptop with left arm`（episode 3123，起始帧 212）
- `pick up the closed laptop`（episode 3124，起始帧 259）
- `retract arms to home positions`（episode 3124，起始帧 544）
- `hand over laptop from left to right arm`（episode 3126，起始帧 408）
- `pick up the headphones with the right gripper`（episode 3129，起始帧 0）
- `handover the headphones from right to left gripper`（episode 3129，起始帧 36）
- `handover the laptop from right to left arm`（episode 3129，起始帧 485）
- `place the laptop into the vertical stand`（episode 3129，起始帧 605）
- `retract the left arm`（episode 3129，起始帧 731）
- `retract right gripper`（episode 3131，起始帧 512）
- `handover laptop from right to left arm`（episode 3138，起始帧 192）
- `close the laptop lid with right arm`（episode 3142，起始帧 108）
- `pick up the closed laptop with left arm`（episode 3142，起始帧 210）
- `handover laptop from left arm to right arm`（episode 3142，起始帧 302）
- `place the laptop into the vertical stand with right arm`（episode 3142，起始帧 386）
- `handover the laptop`（episode 3151，起始帧 228）
- `place the laptop in the stand`（episode 3151，起始帧 323）
- `close the laptop screen`（episode 3152，起始帧 84）
- `pick up closed laptop with left arm`（episode 3152，起始帧 179）
- `retract right arm from laptop`（episode 3155，起始帧 453）
- `handover laptop from right to left gripper`（episode 3158，起始帧 222）
- `retract arms`（episode 3158，起始帧 425）
- `place laptop in the stand`（episode 3160，起始帧 369）
- `handover laptop to right gripper`（episode 3162，起始帧 291）
- `retract`（episode 3164，起始帧 478）
- `retract both arms`（episode 3165，起始帧 473）
- `place laptop into the vertical stand`（episode 3177，起始帧 367）
- `handover laptop from left to right hand`（episode 3186，起始帧 325）
- `retract right arm to home position`（episode 3186，起始帧 493）
- `retract right arm and approach laptop with left arm`（episode 3195，起始帧 358）

### Task 32（1 条）

L3：Pick up the two T-shaped blocks, swap their positions, and place them back with the correct orientations.

- `Pick up the two T-shaped blocks, swap their positions, and place them back with the correct orientations.`（episode 3200，起始帧 0）

### Task 33（16 条）

L3：Swap the two blocks using the empty mat, pressing the button after each move.

- `pick up the first block from the second pad`（episode 3300，起始帧 0）
- `place the first block on the first pad`（episode 3300，起始帧 36）
- `press the button after the first move`（episode 3300，起始帧 78）
- `pick up the second block from the third pad`（episode 3300，起始帧 140）
- `place the second block on the second pad`（episode 3300，起始帧 178）
- `press the button after the second move`（episode 3300，起始帧 220）
- `pick up the first block from the first pad`（episode 3300，起始帧 277）
- `place the first block on the third pad`（episode 3300，起始帧 315）
- `press the button after the third move`（episode 3300，起始帧 361）
- `retract both arms`（episode 3300，起始帧 421）
- `pick up the first block from the third pad`（episode 3301，起始帧 0）
- `place the first block on the second pad`（episode 3301，起始帧 35）
- `pick up the second block from the first pad`（episode 3301，起始帧 131）
- `place the second block on the third pad`（episode 3301，起始帧 168）
- `pick up the second block from the second pad`（episode 3303，起始帧 141）
- `place the second block on the first pad`（episode 3305，起始帧 181）

### Task 34（5 条）

L3：Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.

- `pick up the broom with the left arm`（episode 3400，起始帧 0）
- `handover the broom from the left arm to the right arm`（episode 3400，起始帧 76）
- `pick up the dustpan with the left arm`（episode 3400，起始帧 174）
- `sweep the blocks into the dustpan using the broom in the right arm`（episode 3400，起始帧 323）
- `retract both arms`（episode 3400，起始帧 686）
