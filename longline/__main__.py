"""Allow running longline as a package: python -m longline."""

# 包入口点：支持通过 `python -m longline` 方式运行本项目
from longline.main import main

# 直接调用主函数启动 CLI
main()
