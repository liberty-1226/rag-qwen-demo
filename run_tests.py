import requests
import json

# 加载测试用例
with open('test_cases.json', 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

base_url = "http://127.0.0.1:8000/ask"

print("🚀 开始运行 RAG 系统自动化测试...\n")

for case in test_cases:
    q = case['question']
    expected = case['expected_answer']
    
    # 发送请求
    try:
        response = requests.post(base_url, json={"question": q}, timeout=5)
        answer = response.json().get('answer', '')
        
        # 简单评估：如果预期答案在返回答案中，则算通过
        status = "✅ 通过" if expected in answer else "❌ 失败"
        
        print(f"[{case['category']}] 问题: {q}")
        print(f"   预期: {expected}")
        print(f"   实际: {answer[:50]}...")
        print(f"   结果: {status}\n")
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}\n")

print("🏁 测试完成！")