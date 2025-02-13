import os
import json
import sys
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
sys.path.append(root_path)

from fep_nbv.utils.utils import offset2word

def record_remaining_models(shapenet_dir, output_json):
    """
    扫描指定文件夹中的剩余模型，并保存到 JSON 文件。
    
    :param shapenet_dir: ShapeNet 文件夹路径
    :param output_json: 保存剩余模型路径的 JSON 文件
    """
    remaining_models = {}

    for category in os.listdir(shapenet_dir):
        category_path = os.path.join(shapenet_dir, category)
        if not os.path.isdir(category_path):
            continue
        category_name = offset2word(category)
        
        # 获取模型子文件夹
        model_folders = [os.path.join(category, model) for model in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, model))]
        remaining_models[category] = model_folders
        print(f'{category_name} \t\t\t remain {len(model_folders)} models')
        # print(f'{category_name}')

    # 保存到 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(remaining_models, f, indent=4)
    print(f"模型路径已保存到 {output_json}")


def delete_removed_models(full_shapenet_dir, remaining_json):
    """
    根据 JSON 文件删除完整 ShapeNet 数据集中的未保留模型。
    
    :param full_shapenet_dir: 完整的 ShapeNet 数据集路径
    :param remaining_json: 保存剩余模型路径的 JSON 文件
    """
    # 加载剩余模型记录
    with open(remaining_json, 'r') as f:
        remaining_models = json.load(f)

    # 遍历完整数据集并删除未保留的模型
    for category in os.listdir(full_shapenet_dir):
        category_path = os.path.join(full_shapenet_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # 获取该类别下所有模型
        all_models = os.listdir(category_path)
        remaining_in_category = set(model.split('/')[-1] for model in remaining_models.get(category, []))
        
        for model in all_models:
            model_path = os.path.join(category_path, model)
            if model not in remaining_in_category:
                # 删除模型文件夹
                if os.path.isdir(model_path):
                    os.system(f'rm -rf "{model_path}"')
                    print(f"已删除: {model_path}")

    print("未保留的模型已全部删除。")

def merge_json_list_intersection(file1, file2, output_file):
    # 读取第一个 JSON 文件
    with open(file1, 'r') as f:
        dict1 = json.load(f)

    # 读取第二个 JSON 文件
    with open(file2, 'r') as f:
        dict2 = json.load(f)

    # 取两个字典的键的交集
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # 生成新的字典，对于相同键的值，取列表的交集
    merged_dict = {key: list(set(dict1[key]) & set(dict2[key])) for key in common_keys}
    for category in merged_dict.keys():
        category_name = offset2word(category)
        print(f'{category_name} \t\t\t remain {len(merged_dict[category])} models')
        # print(f'{category_name}')

    # 保存到新的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(merged_dict, f, indent=4)

    print(f"合并完成，交集 JSON 文件已保存到: {output_file}")

if __name__=='__main__':
    shapenet_dir = shapenet_path
    output_json = os.path.join(root_path,'json_files/current_shapenet_fudan113.json')
    # delete_removed_models(shapenet_dir, output_json)
    record_remaining_models(shapenet_dir, output_json)

    # file1 = '/home/zhengquan/04-fep-nbv/json_files/current_shapenet_fudan113.json'
    # file2 = '/home/zhengquan/04-fep-nbv/json_files/current_shapenet_ntu.json'
    # merge_json_list_intersection(file1,file2,output_json)

