1. 环境安装
- pip intall -e .

2. 服务启动
- bf16: python -m minisgl
- int8: python -m minisgl --quant int8

3. 精度测试
- python evaluate_accuracy.py

4. 性能测试
- python benchmark/online/bench_simple.py