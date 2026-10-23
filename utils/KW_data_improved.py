import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


class SyncImageAugmenter:
    def __init__(self, img_dir, label_dir, output_suffix="_aug"):
        """初始化同步图像增强器"""
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_suffix = output_suffix

        # 确保输入目录存在
        for dir_path in [img_dir, label_dir]:
            if not os.path.exists(dir_path):
                raise ValueError(f"目录不存在: {dir_path}")

        # 支持的图片格式
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

        # 记录处理的图片数量
        self.processed_pairs = 0
        self.generated_pairs = 0

    def get_matching_pairs(self):
        """获取原图和标签图的匹配对（文件名相同的图片）"""
        img_files = [f for f in os.listdir(self.img_dir)
                     if f.lower().endswith(self.supported_formats) and self.output_suffix not in f]

        matching_pairs = []

        for img_file in img_files:
            label_file = img_file
            label_path = os.path.join(self.label_dir, label_file)

            if os.path.exists(label_path) and os.path.isfile(label_path):
                if self.output_suffix not in label_file:
                    matching_pairs.append((img_file, label_file))

        return matching_pairs

    def save_pair(self, img, label, original_filename, operation_id, operation_name):
        """保存增强后的原图和标签图，添加序号确保唯一性"""
        base_name, ext = os.path.splitext(original_filename)
        # 添加序号确保即使操作相同也能生成不同文件名
        new_img_name = f"{base_name}{self.output_suffix}_{operation_id}_{operation_name}{ext}"
        new_label_name = f"{base_name}{self.output_suffix}_{operation_id}_{operation_name}{ext}"

        img_path = os.path.join(self.img_dir, new_img_name)
        img.save(img_path)

        label_path = os.path.join(self.label_dir, new_label_name)
        label.save(label_path)

        self.generated_pairs += 1
        return img_path, label_path

    # 增强操作保持不变
    def random_rotation(self, img, label):
        angles = [-30, -20, -15, -10, -5, 5, 10, 15, 20, 30]
        angle = random.choice(angles)
        return (img.rotate(angle, expand=True),
                label.rotate(angle, expand=True),
                f"rot{angle}")

    def random_flip(self, img, label):
        flip_type = random.choice(['horizontal', 'vertical'])
        if flip_type == 'horizontal':
            return (ImageOps.mirror(img), ImageOps.mirror(label), "flip_h")
        else:
            return (ImageOps.flip(img), ImageOps.flip(label), "flip_v")

    def random_zoom(self, img, label):
        zoom_factor = random.uniform(0.8, 1.2)
        width, height = img.size

        img_zoomed = img.resize((int(width * zoom_factor), int(height * zoom_factor)), Image.LANCZOS)
        label_zoomed = label.resize((int(width * zoom_factor), int(height * zoom_factor)), Image.LANCZOS)

        if zoom_factor > 1.0:
            left = (img_zoomed.width - width) // 2
            top = (img_zoomed.height - height) // 2
            right = left + width
            bottom = top + height
            img_zoomed = img_zoomed.crop((left, top, right, bottom))
            label_zoomed = label_zoomed.crop((left, top, right, bottom))
        elif zoom_factor < 1.0:
            left = (width - img_zoomed.width) // 2
            top = (height - img_zoomed.height) // 2

            new_img = Image.new('RGB', (width, height), (0, 0, 0))
            new_label = Image.new(label.mode, (width, height), (0, 0, 0))

            new_img.paste(img_zoomed, (left, top))
            new_label.paste(label_zoomed, (left, top))

            img_zoomed = new_img
            label_zoomed = new_label

        return img_zoomed, label_zoomed, f"zoom{zoom_factor:.1f}"

    def process_pairs(self, num_augmentations=10):
        """处理所有匹配的图片对，确保生成指定数量的增强对"""
        matching_pairs = self.get_matching_pairs()

        if not matching_pairs:
            print("没有找到匹配的原图和标签图对")
            return

        # 定义增强操作列表
        augmentation_functions = [
            self.random_rotation,
            self.random_flip,
            self.random_zoom
        ]

        print(f"找到 {len(matching_pairs)} 对匹配的图片，开始处理...")
        print(f"每对图片将生成 {num_augmentations} 个增强版本")

        for img_file, label_file in matching_pairs:
            try:
                img_path = os.path.join(self.img_dir, img_file)
                label_path = os.path.join(self.label_dir, label_file)

                img = Image.open(img_path).convert('RGB')
                label = Image.open(label_path)

                self.processed_pairs += 1

                # 强制生成指定数量的增强图，使用序号确保不重复
                for i in range(num_augmentations):
                    # 循环选择操作，确保能生成足够数量
                    func = augmentation_functions[i % len(augmentation_functions)]
                    # 对同一种操作再加入随机因素，避免完全重复
                    augmented_img, augmented_label, op_name = func(img.copy(), label.copy())

                    # 保存时加入序号，确保文件名唯一
                    self.save_pair(augmented_img, augmented_label, img_file, i, op_name)

                    if self.generated_pairs % 10 == 0:
                        print(f"已生成 {self.generated_pairs} 对增强图片")

            except Exception as e:
                print(f"处理图片对 {img_file} 时出错: {str(e)}")

        print(f"处理完成！共处理 {self.processed_pairs} 对原始图片，生成 {self.generated_pairs} 对增强图片")


if __name__ == "__main__":
    # 设置原图和标签图的文件夹路径
    image_directory = "C:/Users/DELL/coal_kw_data_yt2"  # 原图文件夹
    label_directory = "C:/Users/DELL/coal_kw_data2"  # 标签图文件夹

    # 创建增强器实例
    augmenter = SyncImageAugmenter(image_directory, label_directory)

    # 设置需要生成的增强图片数量
    augmenter.process_pairs(num_augmentations=10)
