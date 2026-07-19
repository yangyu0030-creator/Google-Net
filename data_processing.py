import os
import shutil
import random

# ==================== 配置区 ====================
train_path = 'data/train'           # 训练集文件夹
test_ratio = 0.2                    # 抽取比例（20%）
seed = 42                           # 随机种子
confirm = True                      # 是否开启确认（True=需要手动输入 y 才继续）
# ===============================================

# ---------- 第一部分：整理散落文件 ----------
if not os.path.exists(train_path):
    print(f'错误：找不到文件夹 "{train_path}"，请检查路径')
    exit()

dog_dir = os.path.join(train_path, 'dog')
cat_dir = os.path.join(train_path, 'cat')
os.makedirs(dog_dir, exist_ok=True)
os.makedirs(cat_dir, exist_ok=True)

for filename in os.listdir(train_path):
    file_path = os.path.join(train_path, filename)
    if not os.path.isfile(file_path):
        continue

    if filename.startswith('dog.'):
        dest = os.path.join(dog_dir, filename)
        if os.path.exists(dest):
            base, ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_name = f"{base}_{counter}{ext}"
                new_dest = os.path.join(dog_dir, new_name)
                if not os.path.exists(new_dest):
                    dest = new_dest
                    break
                counter += 1
        shutil.move(file_path, dest)
        print(f'移动：{filename} -> dog/')

    elif filename.startswith('cat.'):
        dest = os.path.join(cat_dir, filename)
        if os.path.exists(dest):
            base, ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_name = f"{base}_{counter}{ext}"
                new_dest = os.path.join(cat_dir, new_name)
                if not os.path.exists(new_dest):
                    dest = new_dest
                    break
                counter += 1
        shutil.move(file_path, dest)
        print(f'移动：{filename} -> cat/')
    else:
        print(f'忽略：{filename}（前缀不是 dog 或 cat）')

print('整理完成！')

# ---------- 第二部分：划分测试集（关键改动：移动而非复制） ----------
print('\n开始划分测试集...')

# 检查测试集目录是否已存在，如果存在则警告
parent_dir = os.path.dirname(train_path)
test_root = os.path.join(parent_dir, 'test')
if os.path.exists(test_root):
    print(f'⚠️ 警告：测试集目录 "{test_root}" 已存在！')
    if confirm:
        choice = input('是否继续？继续将覆盖原有测试集，且训练集文件会被移动，确认？(y/n): ')
        if choice.lower() != 'y':
            print('已取消操作。')
            exit()
    # 删除旧的测试集目录（避免旧文件和移动的新文件混在一起）
    shutil.rmtree(test_root)
    print('已删除旧测试集目录。')

# 重新创建测试集根目录
os.makedirs(test_root, exist_ok=True)

classes = ['dog', 'cat']
random.seed(seed)

for cls in classes:
    src_dir = os.path.join(train_path, cls)          # data/train/dog
    if not os.path.isdir(src_dir):
        print(f'警告：{src_dir} 不存在，跳过')
        continue

    all_files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f))
                 and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    if not all_files:
        print(f'警告：{src_dir} 中没有图片文件，跳过')
        continue

    random.shuffle(all_files)
    num_test = int(len(all_files) * test_ratio)
    test_files = all_files[:num_test]

    dst_dir = os.path.join(test_root, cls)
    os.makedirs(dst_dir, exist_ok=True)

    # 移动（不是复制）—— 从训练集搬到测试集
    for filename in test_files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 处理重名（虽然理论上不会重名，但保险）
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_name = f"{base}_{counter}{ext}"
                new_dst = os.path.join(dst_dir, new_name)
                if not os.path.exists(new_dst):
                    dst_path = new_dst
                    break
                counter += 1

        # 关键：移动文件（shutil.move），源文件消失
        shutil.move(src_path, dst_path)
        print(f'移动（从训练集移除）：{filename} -> test/{cls}/')

    print(f'{cls} 完成，共移动 {len(test_files)} 张（占 {test_ratio*100:.0f}%）')

print(f'\n✅ 测试集划分完成！')
print(f'训练集路径：{train_path}（已移除测试集图片）')
print(f'测试集路径：{test_root}')
print('现在训练集和测试集完全没有重合。')