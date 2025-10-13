import json

# 读取 JSON 文件
with open("medmcqa_edit_gemini_2.0_flash_final_new.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 用字典存储 prompt 和 locality_prompt 出现的位置
prompt_dict = {}
locality_dict = {}

cnt=0
# 检查 prompt 是否重复
for idx, record in enumerate(data):
    prompt = record.get("prompt")
    if prompt:
        if prompt in prompt_dict:
            cnt+=1
            if cnt<2:
                print(f"记录 {idx} 的 prompt 与 记录 {prompt_dict[prompt]} 的 prompt 相同: {prompt}")
        else:
            prompt_dict[prompt] = idx

print(f"重复的 prompt 总数: {cnt}")

cnt=0
# 检查 locality_prompt 是否重复
for idx, record in enumerate(data):
    loc_prompt = record.get("locality_prompt")
    if loc_prompt:
        if loc_prompt in locality_dict:
            cnt+=1
            if cnt<2:
                print(f"记录 {idx} 的 locality_prompt 与 记录 {locality_dict[loc_prompt]} 的 locality_prompt 相同: {loc_prompt}")
        else:
            locality_dict[loc_prompt] = idx

print(f"重复的 locality_prompt 总数: {cnt}")

locality_map = {}
for idx, record in enumerate(data):
    loc_prompt = record.get("locality_prompt")
    if loc_prompt:
        if loc_prompt not in locality_map:
            locality_map[loc_prompt] = []
        locality_map[loc_prompt].append(idx)

cnt=0
# 检查 prompt 是否出现在其他记录的 locality_prompt 中
for idx, record in enumerate(data):
    prompt = record.get("prompt")
    if prompt and prompt in locality_map:
        for match_idx in locality_map[prompt]:
            if match_idx != idx:  # 跨记录
                cnt+=1
                if cnt<2:
                    print(f"记录 {idx} 的 prompt 与 记录 {match_idx} 的 locality_prompt 相同: {prompt}")
print(f"prompt 出现在其他记录的 locality_prompt 中的总次数: {cnt}")