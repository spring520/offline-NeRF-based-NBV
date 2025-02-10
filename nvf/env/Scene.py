from nvf.env.BlenderRenderer import *
from nvf.env.BlenderProcess import BlenderProcess
import nvf.env.gen_data_fn as gen_data_fn
import contextlib

class HubbleScene(BlenderGLB):
    """docstring for HubbleScene"""
    name = 'hubble'
    def __init__(self, cfg):
        super(HubbleScene, self).__init__(cfg)
        self.hdri_rotation = [0.0, 0.0, 0.0] if not hasattr(cfg, 'hdri_rotation') else cfg.hdri_rotation # wRi
        # self.hdri_path = os.path.join(cfg.root,"assets/hdri/RenderCrate-HDRI_Orbital_38_4K.hdr")
        self.hdri_path = os.path.join(cfg.root,"assets/hdri/gray_hdri.exr")

        # self.hdri_path = "assets/hdri/neon_photostudio_4k.hdr"
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
        self.add_camera()
        self.add_object('hubble', os.path.join(cfg.root,'assets/models/Hubble.glb'), scale=self.cfg.scale)
        self.config_camera()
        self.config_blender()
        self.init_rgb_world()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.hubble_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.hubble_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.hubble_full(scale=cfg.scale),
            'part': lambda :gen_data_fn.hubble_full(Tmax=20, scale=cfg.scale, pxpy=True),
        }
        # self.set_world('rgb')

    def get_mesh(self):
        mesh = trimesh.load(os.path.join(self.cfg.root,'assets/models/Hubble.glb'),force='mesh')
        return mesh 

    def set_white_background(self):
        # 设置世界背景颜色为白色
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        bg_node = world.node_tree.nodes.get('Background')
        
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA白色背景
        
        # 如果使用的是渲染引擎为 Eevee，确保将背景设置为透明
        bpy.context.scene.render.film_transparent = False
    
    def add_cone_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        # quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cone = self.objects.get(f'Cone_{id-1}')
            if cone:
                mat = cone.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone

    def add_cone_at_pose2(self, position, quaternion,id=0, color=(1, 0, 0, 1)):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        # quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        
        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = color  # 红色
        # bsdf.inputs['Roughness'].default_value = 1  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        # bsdf.inputs['Specular'].default_value = 1.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone


    def add_cylinder_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cylinder = self.objects.get(f'Cylinder_{id-1}')
            if cylinder:
                mat = cylinder.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.5, location=(0, 0, 0))
        cylinder = bpy.context.object
        cylinder.name = f'ViewCylinder_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cylinder.data.materials.append(mat)
        
        # 设置圆锥的位置
        cylinder.location = mathutils.Vector(position)
        import math
        cylinder.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cylinder["inst_id"] = inst_id
        self.objects[f'Cylinder_{id}'] = cylinder

    def add_transparent_cube(self, aabb, cube_color=(1,0,0,0.5),edge_color=(1, 0, 0, 1), edge_width=0.5):
        min_corner = aabb[0, :].tolist()
        max_corner = aabb[1, :].tolist()

        # 计算立方体的中心和尺寸
        center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
        size = [(max_corner[i] - min_corner[i]) for i in range(3)]

        # 添加立方体
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        cube = bpy.context.object
        cube.scale = [s / 2 for s in size]  # 调整立方体的尺寸

        # 创建材质
        mat = bpy.data.materials.new(name="TransparentMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = cube_color  # 设置为半透明
        bsdf.inputs['Alpha'].default_value = cube_color[-1]  # 设置为半透明
        mat.blend_method = 'BLEND'  # 设置混合模式为混合

        # 将材质赋予对象
        cube.data.materials.append(mat)

        # 添加 Wireframe 修饰符以创建边框
        bpy.ops.object.modifier_add(type='WIREFRAME')
        cube.modifiers["Wireframe"].thickness = 0.05
        cube.modifiers["Wireframe"].use_replace = False  # 保留原始几何体

        # 创建一个新的材质用于棱边
        edge_mat = bpy.data.materials.new(name="EdgeMaterial")
        edge_mat.use_nodes = True
        edge_bsdf = edge_mat.node_tree.nodes.get("Principled BSDF")
        edge_bsdf.inputs['Base Color'].default_value = edge_color  # 设置棱边颜色
        edge_bsdf.inputs['Alpha'].default_value = 1.0  # 不透明
        edge_mat.blend_method = 'BLEND'  # 设置混合模式为混合

        # 将棱边材质赋予对象
        if len(cube.data.materials) < 2:
            cube.data.materials.append(edge_mat)
        else:
            cube.data.materials[1] = edge_mat

        # 设置对象的显示模式为线框
        cube.display_type = 'WIRE'
        
        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cube["inst_id"] = inst_id
        self.objects[f'Cube_{id}'] = cube

    def add_coordinate_axes(self, location=(0, 0, 0), size=1.0):
        """
        在场景中添加坐标轴。

        Args:
            location (tuple): 坐标轴的位置，默认为原点 (0, 0, 0)。
            size (float): 坐标轴的大小，默认为 1.0。
        """
        bpy.ops.object.empty_add(type='ARROWS', location=location)
        axes = bpy.context.object
        axes.scale = (size, size, size)
         

class LegoScene(BlenderFile):
    """docstring for LegoScene"""
    name = 'lego'
    def __init__(self, cfg):
        super(LegoScene, self).__init__(cfg)
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/lego.blend'))
        self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)

        # self.config_camera()
        # self.config_blender()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.lego_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.lego_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.lego_full(scale=cfg.scale),
        }
        self.get_blender_mesh = super().get_mesh

    def get_mesh(self):
        mesh = trimesh_load_mesh(os.path.join(self.cfg.root, 'assets/blend_files/lego.ply'))
        
        return mesh
    
    def obj_filter(self, obj):
        if obj.name.startswith('Plane'):
            # breakpoint()
            return True
        return False
    
    def render(self):
        # with contextlib.redirect_stderr(open(os.devnull, "w")):
        with stdout_redirected():
            self.load_scene(os.path.join(self.cfg.root, 'assets/blend_files/lego.blend'))
            # self.load_scene(os.path.join(self.cfg.root, 'assets/blend_files/lego.blend'))
            self.set_camera_pose(self.camera_pose)
        return super(LegoScene, self).render()

    def set_white_background(self):
        # 设置世界背景颜色为白色
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        bg_node = world.node_tree.nodes.get('Background')
        
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA白色背景
        
        # 如果使用的是渲染引擎为 Eevee，确保将背景设置为透明
        bpy.context.scene.render.film_transparent = False
    
    def add_cone_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cone = self.objects.get(f'Cone_{id-1}')
            if cone:
                mat = cone.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone

    def add_cylinder_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cylinder = self.objects.get(f'Cylinder_{id-1}')
            if cylinder:
                mat = cylinder.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.5, location=(0, 0, 0))
        cylinder = bpy.context.object
        cylinder.name = f'ViewCylinder_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cylinder.data.materials.append(mat)
        
        # 设置圆锥的位置
        cylinder.location = mathutils.Vector(position)
        import math
        cylinder.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cylinder["inst_id"] = inst_id
        self.objects[f'Cylinder_{id}'] = cylinder

class HotdogScene(BlenderFile):
    """docstring for DrumsScene"""
    name = 'hotdog'
    def __init__(self, cfg):
        super(HotdogScene, self).__init__(cfg)
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/hotdog.blend'))
        # self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)
        self.postprocess_fn = None

        self.config_camera()
        self.config_blender()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.hotdog_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.hotdog_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.hotdog_full(scale=cfg.scale),
            'full2': lambda :gen_data_fn.hotdog_full(Tmax=20, scale=cfg.scale),
            'part': lambda :gen_data_fn.hotdog_full(Tmax=20, scale=cfg.scale, pxpy=True),
        }
        self.objects = {}
    
    def get_mesh(self):
        # with open('assets/blend_files/hotdog.obj', 'r') as f:
        mesh = trimesh_load_mesh(os.path.join(self.cfg.root, 'assets/blend_files/hotdog.ply'))
        
        return mesh
    
    def set_white_background(self):
        # 设置世界背景颜色为白色
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        bg_node = world.node_tree.nodes.get('Background')
        
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA白色背景
        
        # 如果使用的是渲染引擎为 Eevee，确保将背景设置为透明
        bpy.context.scene.render.film_transparent = False
    
    def add_cone_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cone = self.objects.get(f'Cone_{id-1}')
            if cone:
                mat = cone.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone

    def add_cylinder_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cylinder = self.objects.get(f'Cylinder_{id-1}')
            if cylinder:
                mat = cylinder.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.5, location=(0, 0, 0))
        cylinder = bpy.context.object
        cylinder.name = f'ViewCylinder_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cylinder.data.materials.append(mat)
        
        # 设置圆锥的位置
        cylinder.location = mathutils.Vector(position)
        import math
        cylinder.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cylinder["inst_id"] = inst_id
        self.objects[f'Cylinder_{id}'] = cylinder



class RoomScene(BlenderFile):
    """docstring for RoomScene"""
    name = 'room'
    def __init__(self, cfg):
        super(RoomScene, self).__init__(cfg)
        self.cfg.n_init_views = 10
        self.set_mesh_filename(filepath=os.path.join(cfg.root, 'assets/blend_files/room.obj'))
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/room.blend'))
        self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)
        self.gen_data_fn = {
            'init': lambda :gen_data_fn.room_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.room_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.room_full(scale=cfg.scale),
            'part': lambda :gen_data_fn.room_part(scale=cfg.scale),
        }
        self.config_camera()
        self.config_blender()

        self.valid_range = np.array([[-2.,-2.,-2.], [2.,2.,2.]])*20

        self.objects = {}

    def get_mesh(self):
        # Get all objects in the scene
        filename = self.get_mesh_filename()
        mesh = trimesh.load(filename,force='mesh')
        TF=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        mesh = mesh.apply_transform(TF)
        return mesh
        
    def set_white_background(self):
        # 设置世界背景颜色为白色
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        bg_node = world.node_tree.nodes.get('Background')
        
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA白色背景
        
        # 如果使用的是渲染引擎为 Eevee，确保将背景设置为透明
        bpy.context.scene.render.film_transparent = False
    
    def add_cone_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cone = self.objects.get(f'Cone_{id-1}')
            if cone:
                mat = cone.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone

    def add_cylinder_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cylinder = self.objects.get(f'Cylinder_{id-1}')
            if cylinder:
                mat = cylinder.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.5, location=(0, 0, 0))
        cylinder = bpy.context.object
        cylinder.name = f'ViewCylinder_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cylinder.data.materials.append(mat)
        
        # 设置圆锥的位置
        cylinder.location = mathutils.Vector(position)
        import math
        cylinder.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cylinder["inst_id"] = inst_id
        self.objects[f'Cylinder_{id}'] = cylinder
    
def trimesh_load_mesh(path):
    mesh_or_scene = trimesh.load(path)

    # if self.cfg.scale !=1.:
    #     matrix = np.eye(4)
    #     matrix[:2, :2] *= self.cfg.scale
    #     mesh_or_scene.apply_transform(matrix)

    if isinstance(mesh_or_scene, trimesh.Scene):
        # If the scene contains multiple meshes, you have a few options:
        
        # Option 1: Get a single mesh (if you know there's only one)
        if len(mesh_or_scene.geometry) == 1:
            mesh = next(iter(mesh_or_scene.geometry.values()))

        # Option 2: Combine all meshes into one
        else:
            mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        # The loaded object is already a Trimesh object
        mesh = mesh_or_scene
    return mesh


