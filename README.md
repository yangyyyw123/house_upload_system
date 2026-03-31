# house_upload_system

这个仓库现在按“源码进 Git，运行产物留本地”的方式管理。

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
