# MultiCamera4RC v0.2.0

这是当前项目的独立源码发布版本，来源为 `bestzbf/my_dfrt` 的 `codex/multicamera-4rc` 分支、提交 `404d4fdfe547574bb2cbfa6a283a1dae917c7a15`。新仓库默认分支为 `main`，版本标签为 `v0.2.0`。

## 发布范围

完整保留源提交的 658 个受 Git 跟踪文件，包括 MultiCamera4RC 模型、训练和推理、混合数据加载器、Syn4D 适配、GT 审计、验证控制器、单目评估、测试、文档，以及原有 D4RT 参考代码。每个来源文件的 SHA256 见 [source_provenance.json](source_provenance.json)。

发布整理只更新根 README 的仓库/安装说明及 `pyproject.toml` 的发行包名称和仓库链接，并新增本版本的说明、源码打包清单、来源记录与验证证明；模型、损失、GT、训练和评估算法保持源版本。历史审计中的旧提交、分支及服务器路径作为证据保留。

Python distribution 由 `my-dfrt-multicamera4rc` 改为 `multicamera4rc`；Python import 和命令仍为 `multicamera4rc`，版本为 `0.2.0`。源码包包含完整仓库；wheel 包含当前可安装模型包。仓库级 `scripts/` 和历史 D4RT 入口请从源码 checkout 中运行。

这是代码发布，训练数据与 giant checkpoint 由用户自行准备；真实训练数据路径需按 [数据文档](../../docs/data.md) 和 [训练文档](../../docs/training.md) 配置。第三方许可及上游来源保留，见 [NOTICE](../../NOTICE)。

## 安装

```bash
git clone --branch v0.2.0 https://github.com/bestzbf/MultiCamera4RC.git
cd MultiCamera4RC
python -m venv .venv
source .venv/bin/activate
# 先安装符合本机 CPU/CUDA 环境的 PyTorch，再安装项目。
python -m pip install -e '.[data,test,weights]'
python -m pytest -q
python -m multicamera4rc demo --output outputs/release_smoke --train-steps 2
```

GitHub Release 提供完整源码压缩包、Python wheel、校验和及本次验证记录；也可从 GitHub 的版本标签下载源码。wheel 可用 `python -m pip install '/path/to/multicamera4rc-0.2.0-py3-none-any.whl[data,weights]'` 安装。

## 已有实测结果

[55.5k 单目评估](../../multicamera4rc/audits/monocular_benchmark_20261008/README.md)已完整覆盖 ScanNet test100 和 Sintel Final 14，零失败。Depth S/SS AbsRel 分别为 ScanNet `0.020179 / 0.017680`、Sintel `0.255322 / 0.202269`。评估输入保持比例、最长边 518，评分网格 256×256；当前相机 head 与旧 D4RT 的查询位姿估计不同。该评估针对几何输出，不覆盖 motion decoder 的 scene flow / 长期追踪。

## 本次发布验证

实测 **372 项测试通过，0 个失败、0 个跳过**。wheel 在新的虚拟环境中安装，从源码目录之外执行 CLI help、tiny 训练/恢复/推理及原生 adapter demo；全部通过。安装测试复用本机已安装的第三方依赖，`multicamera4rc` 模块实际从新环境中的 wheel 加载。

详情见 [validation.json](validation.json)。另核验所有来源 Python 文件未变、源码包完整覆盖 658 个原始文件、wheel 中配置与上游许可保留，以及 README 相对链接。
