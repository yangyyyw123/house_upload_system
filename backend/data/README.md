`backend/data/` 存放数据层代码：

- `db.py`：SQLAlchemy 实例
- `models.py`：数据库模型
- `bootstrap.py`：建表与 SQLite 兼容补列
- `queries.py`：房屋、记录、靶标等数据访问与绑定辅助函数

目标是让 `backend/app.py` 不再直接承载模型定义和数据库结构修正逻辑。
