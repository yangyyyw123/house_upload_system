# house_upload_system

这个仓库现在按“源码进 Git，运行产物留本地”的方式管理。

## 当前目录结构

```text
house_upload_system/
├─ backend/
│  ├─ app.py
│  ├─ core/                 # 配置、常量、时间与运行时公共层
│  ├─ data/                 # 数据库实例、模型与建表修正逻辑
│  ├─ services/
│  │  ├─ crack/             # 裂缝分析服务主体
│  │  ├─ targets/           # 靶标生成服务主体
│  │  └─ *.py               # 兼容入口，转发到更深层实现
│  ├─ templates/
│  │  ├─ pages/
│  │  │  ├─ platform/       # 平台首页模板正文
│  │  │  └─ targets/        # 靶标详情页模板正文
│  │  ├─ index.html         # 兼容入口，转发到 pages/index.html
│  │  └─ target_detail.html # 兼容入口，转发到 pages/target_detail.html
│  └─ uploads/              # 本地运行产物
├─ frontend/
│  ├─ assets/
│  │  └─ branding/          # 校徽等品牌图片
│  ├─ styles/
│  │  └─ platform.css       # 前端样式主体
│  ├─ style.css             # 前端样式兼容入口
│  └─ script.js             # 前端脚本入口
├─ scripts/
│  └─ windows/              # Windows 启动脚本主体
├─ docs/
│  └─ archive/              # 临时文档归档
├─ tests/
│  └─ fixtures/
│     └─ images/            # 测试示例图片
├─ utils/
│  ├─ analysis/             # 分析与量化脚本
│  ├─ image_ops/            # 图像切分与拼接脚本
│  ├─ ui/                   # 桌面查看器组件
│  ├─ notebooks/            # 实验 notebook
│  └─ *.py                  # 兼容入口
├─ model/
└─ start_backend.ps1
```

## 兼容性说明

- `backend/app.py` 的运行入口保持不变。
- `backend/core/` 承载配置、常量、时间处理和运行时辅助逻辑，`app.py` 继续保留主业务编排。
- `frontend/style.css`、`frontend/script.js` 的静态入口保持不变。
- 原 `backend/*.py` 服务文件名保留为兼容层，实际实现已经整理到 `backend/services/crack/`、`backend/services/targets/`。
- 原 `backend/templates/index.html` 和 `backend/templates/target_detail.html` 保留为兼容模板壳，实际页面内容在 `backend/templates/pages/platform/`、`backend/templates/pages/targets/`。
- 根目录 `start_backend.ps1`、`start_backend.bat` 保留为兼容启动入口，实际脚本主体在 `scripts/windows/`。

## Git 管理规则

- 提交到 GitHub：`backend/`、`frontend/`、`utils/`、依赖清单、配置说明。
- 不提交到 GitHub：虚拟环境、数据库、上传图片、分割结果、模型权重。
- 模型参数文件 `model/*_hyperparams.json` 可以保留在仓库里。
- 模型权重 `model/*.pth` 只保留本地；如果以后一定要放 GitHub，建议改用 Git LFS 或 Release 附件。

## 关于“修改前 / 修改后”

不需要手动保存两份同名代码文件。Git 提交历史天然就是“修改前”和“修改后”：

1. 修改前先提交一次。
2. 改完后再提交一次。
3. 在 GitHub 或本地 `git diff` 中查看两个版本差异。

## 首次提交建议

```powershell
git add .
git commit -m "Initial project import"
git push -u origin main
```

如果默认分支不是 `main`，把最后一行里的分支名替换成实际分支。
