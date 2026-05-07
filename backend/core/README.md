`backend/core/` 存放不直接承担业务路由的公共层代码：

- `settings.py`：路径、常量、环境配置
- `common.py`：文本、JSON、编码等基础工具
- `runtime.py`：运行时目录、URL、依赖诊断等辅助逻辑
- `time_utils.py`：时间序列化、格式化与记录时间辅助

目标是让 `backend/app.py` 主要承担主流程编排、模型定义和路由组织。
