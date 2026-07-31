# Persona assets

该目录用于存放 8 个 Q 版像素人格结果卡或透明角色素材：

- `clear_tool_user.webp` — 清醒工具人
- `cautious_cyber_apprentice.webp` — 谨慎型赛博学徒
- `half_understood_master.webp` — 半懂型效率大师
- `high_speed_repeater.webp` — 高速复读机
- `fog_operator.webp` — 迷雾操作员
- `silicon_spokesperson.webp` — 硅基代言人
- `multi_model_voter.webp` — 多模投票教徒
- `ai_authorized_know_it_all.webp` — AI 授权懂王

生产版本建议不要让图像模型直接生成最终中文卡片。图像模型只生成无文字角色，程序再根据 `data/personas.json` 和评分结果绘制标题、标签、分数条与行为证据。

当前目录包含与 `data/personas.json` 一一对应的 8 张完整人格卡。
