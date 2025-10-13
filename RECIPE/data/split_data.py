import json
from collections import Counter

# # 读取 JSON 文件
# with open("medmcqa_edit_gemini_2.0_flash_6000.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 合并 subject_name 和 locality_subject_name
# all_subjects = []
# for record in data:
#     if "subject_name" in record and record["subject_name"]:
#         all_subjects.append(record["subject_name"])
#     if "locality_subject_name" in record and record["locality_subject_name"]:
#         all_subjects.append(record["locality_subject_name"])

# # 统计出现次数
# counter = Counter(all_subjects)
# total = sum(counter.values())

# # 打印结果（数量 + 百分比）
# for subject, count in counter.most_common():
#     percentage = count / total * 100
#     print(f"{subject}: {count} ({percentage:.2f}%)")

# print(f"\n总数: {total}")

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
with open("medmcqa_edit_gemini_2.0_flash_6000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 先切 train (3000) vs 剩余 (3000)
train_df, temp_df = train_test_split(
    df,
    train_size=3000,
    stratify=df["subject_name"],
    random_state=42
)

# 再切 valid (2000) vs test (1000)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=1000,
    stratify=temp_df["subject_name"],
    random_state=42
)

print("Train size:", len(train_df))
print("Valid size:", len(valid_df))
print("Test size:", len(test_df))

# 输出分布看看
print("Train distribution:\n", train_df["subject_name"].value_counts(normalize=True))
print("Valid distribution:\n", valid_df["subject_name"].value_counts(normalize=True))
print("Test distribution:\n", test_df["subject_name"].value_counts(normalize=True))

# 保存
train_df.to_json("train.json", orient="records", force_ascii=False, indent=2)
valid_df.to_json("valid.json", orient="records", force_ascii=False, indent=2)
test_df.to_json("test.json", orient="records", force_ascii=False, indent=2)

