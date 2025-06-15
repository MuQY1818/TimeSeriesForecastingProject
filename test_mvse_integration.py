"""
快速测试 MVSE 集成到 T-LAFS 的功能
"""

import pandas as pd
import numpy as np
from clp_probe_experiment import TLAFS_Algorithm

def test_mvse_integration():
    """测试 MVSE 集成功能"""
    print("🧪 测试 MVSE 集成到 T-LAFS...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # 生成带有季节性的时间序列
    t = np.arange(len(dates))
    seasonal = 5 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = np.random.normal(0, 1, len(dates))
    temp = 15 + seasonal + trend + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temp': temp
    })
    
    print(f"📊 测试数据: {len(df)} 个样本")
    
    # 设置必要的类属性
    TLAFS_Algorithm.target_col_static = 'temp'
    
    # 测试 MVSE 操作
    test_plan = [{"operation": "create_mvse_features"}]
    
    try:
        print("\n🔧 测试 execute_plan 中的 MVSE 操作...")
        result_df = TLAFS_Algorithm.execute_plan(df, test_plan)
        
        # 检查 MVSE 特征
        mvse_cols = [col for col in result_df.columns if 'mvse_' in col]
        
        if len(mvse_cols) > 0:
            print(f"✅ MVSE 集成成功！生成了 {len(mvse_cols)} 个特征")
            print(f"   特征列表: {mvse_cols[:5]}{'...' if len(mvse_cols) > 5 else ''}")
            
            # 检查特征质量
            non_null_count = result_df[mvse_cols].notna().sum().sum()
            total_count = len(result_df) * len(mvse_cols)
            print(f"   特征覆盖率: {non_null_count}/{total_count} ({100*non_null_count/total_count:.1f}%)")
            
        else:
            print("❌ MVSE 集成失败：没有生成 MVSE 特征")
            
    except Exception as e:
        print(f"❌ MVSE 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ MVSE 集成测试完成！")

if __name__ == "__main__":
    test_mvse_integration() 