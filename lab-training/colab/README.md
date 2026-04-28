# Google Colab 版本 Notebook

本目录包含所有实操notebook的Google Colab版本，用于课后练习。

## 使用方式

1. 登录 [Google Colab](https://colab.research.google.com/)
2. 上传本目录下的notebook文件，或从GitHub直接打开
3. 在菜单中选择 **运行时 → 更改运行时类型 → T4 GPU**
4. 运行第一个单元格安装依赖

## 与服务器版本的区别

| 项目 | 服务器版本 | Colab版本 |
|------|-----------|-----------|
| 数据加载 | 本地路径 | URL下载/Google Drive |
| GPU | 租用服务器GPU | 免费T4 GPU |
| 运行时长 | 无限制 | 约12小时/session |
| 依赖安装 | 已预装 | notebook首个cell自动安装 |

## 注意事项

- Colab的session会在闲置一段时间后断开，注意及时保存结果
- 免费GPU有一定使用限额，长时间训练建议使用AutoDL等云GPU平台
- 所有notebook的第一个cell包含环境配置代码，请务必先运行
