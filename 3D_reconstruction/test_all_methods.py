#!/usr/bin/env python3
"""
测试所有EVREAL方法是否可以成功加载
"""

import sys
import os
import subprocess
import time

def test_method(method_name, timeout=60):
    """测试单个方法是否可以成功加载"""
    
    print(f"\n{'='*50}")
    print(f"测试方法: {method_name}")
    print(f"{'='*50}")
    
    cmd = [
        'conda', 'run', '-n', 'Umain2', 'python', 'eval.py',
        '-c', 'std',
        '-m', method_name,
        '-d', 'lego',
        '-qm', 'ssim', 'lpips'
    ]
    
    try:
        # 运行命令，只等待10秒看是否能成功开始处理
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd='/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main'
        )
        
        # 等待一段时间看输出
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(line.strip())
                
                # 检查成功指标
                if "it/s]" in line and "%" in line:  # 进度条开始
                    print(f"✅ {method_name}: 成功开始处理图像")
                    process.terminate()
                    return True
                    
                # 检查错误指标  
                if "Exception" in line or "Error" in line or "cannot instantiate" in line:
                    print(f"❌ {method_name}: 发现错误")
                    process.terminate()
                    return False
                    
            elif process.poll() is not None:
                # 进程已结束
                break
                
        # 超时或进程结束
        if process.poll() is None:
            process.terminate()
            print(f"⏰ {method_name}: 超时 ({timeout}s)")
            return False
        else:
            print(f"🏁 {method_name}: 进程提前结束")
            return "Exception" not in " ".join(output_lines)
            
    except Exception as e:
        print(f"❌ {method_name}: 执行异常 - {e}")
        return False

def main():
    """主测试函数"""
    
    print("EVREAL 方法兼容性测试")
    print("检查所有方法是否能成功加载模型并开始处理")
    
    # 所有8个方法
    all_methods = [
        'E2VID', 'E2VID+', 'FireNet', 'FireNet+', 
        'ET-Net', 'HyperE2VID', 'SPADE-E2VID', 'SSL-E2VID'
    ]
    
    # 之前失败的方法
    previously_failed = ['E2VID+', 'FireNet+', 'ET-Net', 'HyperE2VID']
    
    results = {}
    
    print(f"\n测试 {len(all_methods)} 个方法...")
    
    for method in all_methods:
        success = test_method(method, timeout=30)
        results[method] = success
        
        # 给系统一点时间释放资源
        time.sleep(2)
    
    # 输出结果总结
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    successful_methods = []
    failed_methods = []
    
    for method, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        was_previously_failed = "🔧 之前失败" if method in previously_failed else "✅ 之前成功"
        print(f"{method:15} : {status:8} ({was_previously_failed})")
        
        if success:
            successful_methods.append(method)
        else:
            failed_methods.append(method)
    
    print(f"\n总结:")
    print(f"成功: {len(successful_methods)}/8 方法")
    print(f"失败: {len(failed_methods)}/8 方法")
    
    if previously_failed:
        fixed_methods = [m for m in previously_failed if results.get(m, False)]
        print(f"修复成功: {len(fixed_methods)}/4 之前失败的方法")
        
        if len(fixed_methods) == 4:
            print("🎉 所有WindowsPath问题已修复！")
        else:
            print(f"仍需修复: {[m for m in previously_failed if not results.get(m, False)]}")

if __name__ == "__main__":
    main()