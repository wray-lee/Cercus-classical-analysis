# 1. 定义需要遍历的路径数组
$paths = @(
    # 'D:\Data - 副本\+200\2026.6.23 +200',
    # 'D:\Data - 副本\0 180°\2026.6.23 0',
    # 'D:\Data - 副本\0 180°\-2026.6.23 0 2',
    # 'D:\Data - 副本\-119 80°\-2026.6.27 0.625',
    # 'D:\Data - 副本\-225 48°\--2026.6.27 0.613',
    # 'D:\Data - 副本\-261 42°\2026.6.27 0.662',
    # 'D:\Data - 副本\-308 36°\2026.6.27 -308',
    # 'D:\Data - 副本\-373 30°\-2026.6.26 -373',
    # 'D:\Data - 副本\bv\2026.24 bv',
    # 'D:\Data - 副本\bw\2026.6.17 bw',
    # 'D:\Data - 副本\bw\2026.6.23 bw'
    'D:\Data\bv\2026.7.3'
    'D:\Data\bv\2026.7.5'
    ‘D:\Data\bv\2026.7.5 2'
)

# 2. 遍历执行
foreach ($p in $paths) {
    Write-Host "正在处理: $p" -ForegroundColor Green

    python .\main.py --input-dir "$p" --save "$p"
    python .\plot_trial_panels.py --input-dir "$p" --save "$p\trials"
}