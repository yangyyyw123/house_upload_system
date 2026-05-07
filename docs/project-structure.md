# 项目结构说明

本次整理遵循两个原则：

1. 运行入口不变。
2. 新目录优先承载实际内容，旧路径保留兼容层。

## 目录职责

- `backend/app.py`
  Flask 主入口，继续作为启动脚本使用。
- `backend/services/`
  放置后端服务兼容入口和分域目录。
- `backend/core/`
  放置路径配置、常量、时间工具、运行时辅助等公共层代码。
- `backend/data/`
  放置 SQLAlchemy 实例、数据模型和数据库初始化/补列逻辑。
- `backend/data/queries.py`
  放置与房屋、检测记录、靶标绑定相关的数据访问辅助函数。
- `backend/services/crack/`
  放置裂缝分割、裂缝量化等裂缝分析服务主体。
- `backend/services/targets/`
  放置二维码靶标生成相关服务主体。
- `backend/templates/pages/`
  放置页面模板兼容入口和按页面域拆分后的正文目录。
- `backend/templates/pages/platform/`
  放置平台首页模板正文。
- `backend/templates/pages/targets/`
  放置靶标详情页模板正文。
- `backend/templates/*.html`
  兼容模板入口，避免现有 `render_template(...)` 调用失效。
- `frontend/assets/branding/`
  放置校徽、校名锁定图等前端品牌资源。
- `frontend/styles/`
  放置样式主体文件，`frontend/style.css` 保留为兼容入口。
- `scripts/windows/`
  放置 Windows 启动脚本主体，根目录启动脚本保留为兼容入口。
- `docs/archive/`
  放置临时文档和过程性材料，避免根目录堆积。
- `tests/fixtures/images/`
  放置手工测试使用的图片样例。
- `utils/analysis/`
  放置分析与量化相关脚本。
- `utils/image_ops/`
  放置图像切分、拼接等处理脚本。
- `utils/ui/`
  放置桌面查看器组件。
- `utils/notebooks/`
  放置实验 notebook。

## 兼容层约定

- 如果旧路径已经被外部脚本、历史文档或人工操作习惯使用，不直接删除，先保留薄兼容层。
- 新开发优先引用新目录，例如：
  - 公共配置与通用辅助优先维护在 `backend/core/`。
  - 数据模型与数据库初始化优先维护在 `backend/data/`。
  - Python 服务实现优先从 `backend/services/crack/`、`backend/services/targets/` 引用。
  - 模板正文优先维护 `backend/templates/pages/platform/`、`backend/templates/pages/targets/`。
  - 前端品牌资源优先放在 `frontend/assets/branding/`。
  - 前端样式主体优先维护 `frontend/styles/platform.css`。
  - 启动脚本主体优先维护 `scripts/windows/`。
  - 工具脚本优先维护 `utils/analysis/`、`utils/image_ops/`、`utils/ui/`。
