# -*- coding: utf-8 -*-

import os
import gradio as gr
from context import Context

class GradioGUI:
    def __init__(self, methods: list):
        self.MISC = ['使用缓存', '显示模板牙']
        self.POST = ['显示SDF切面', '显示误差分布']
        self.LOCS = ['#11', '#15']
        self.RESOL_MAP = {
            '低 (128x128)': 128,
            '中 (256x256)': 256,
            '高 (512x512)': 512
        }
        self.Methods = methods

        theme = gr.themes.Soft(
            primary_hue="slate",
        ).set(
            checkbox_background_color_dark='*neutral_900'
        )
        self.inst = gr.Blocks(theme=theme)
        self._ctx = None # current session context
        self.__build()

    def __build(self):
        reverse_index = {self.Methods[ind].__str__():ind for ind in range(len(self.Methods))}

        with self.inst:
            gr.Markdown(
                '# 残缺牙齿重建\n'
                '可用方案：**ToothDIT** 和 **DIF-Net**。\n<br/>'
                '从最左侧上传残缺牙齿模型，选择牙齿位置，点击开始重建即可。\n<br/>'
                '重建过程包括：`点云采样` -> `符号距离场重建` -> `网格提取` -> `误差分布计算`。\n<br/>'
                '<span style="color:blue"><strong>不要在任务进行中刷新网页，否则会丢失计算进度。</strong></span>'
            )
            with gr.Row(equal_height=False):
                # Input part
                with gr.Column():
                    gr.Markdown('### 输入区')
                    inp_mesh = gr.Model3D(label='残缺牙齿模型')
                    inp_methods = gr.Dropdown(
                        label='补全方法',
                        choices=[mtd.__str__() for mtd in self.Methods],
                        value=self.Methods[0].__str__(),
                        multiselect=True
                    )
                    inp_loc = gr.Radio(
                        self.LOCS,
                        value=self.LOCS[0],
                        label='牙齿位置',
                    )

                    with gr.Row():
                        inp_sample_up = gr.Slider(1.0, 100.0, value=100.0, step=1.0, label='切端采样比例')
                        inp_sample_mid = gr.Slider(1.0, 100.0, value=100.0, step=1.0, label='中端采样比例')
                        inp_sample_down = gr.Slider(1.0, 100.0, value=100.0, step=1.0, label='龈端采样比例')

                    inp_iter = gr.Slider(300, 1000, step=10, label="重建迭代次数", value=500)
                    inp_resol = gr.Dropdown(
                        label="网格重建精度",
                        info="括号中为Marching Cubes的体素精度",
                        choices=self.RESOL_MAP.keys(),
                        value=list(self.RESOL_MAP.keys())[1], # 256 x 256
                    )

                    with gr.Row():
                        inp_miscs = gr.CheckboxGroup(
                            label='杂项',
                            choices=self.MISC,
                            value=self.MISC[:1]
                        )
                    
                    with gr.Accordion(label='后处理', open=False):
                        inp_smooth = gr.Slider(0.0, 50.0, step=1.0, value=10.0, label='网格光滑程度')
                        with gr.Row():
                            inp_post = gr.CheckboxGroup(
                                show_label=False,
                                choices=self.POST,
                                value=[]
                            )

                    btn_commit = gr.Button('开始重建', interactive=False)

                # Output part
                with gr.Column():
                    gr.Markdown('### 输出区')
                    show_otps = []
                    otp_meshes = []
                    for mtd in self.Methods:
                        with gr.Column(visible=mtd == self.Methods[0]) as show_otp:
                            otp_mesh = gr.Model3D(label=f'完整牙齿模型 ({mtd})')
                            show_otps.append(show_otp)
                            otp_meshes.append(otp_mesh)

                # Misc part
                with gr.Column(visible=False) as show_total:
                    gr.Markdown('### 附加输出区')
                    show_templates, show_slices, show_errors = [], [], []
                    otp_templates, otp_slices, otp_errors = [], [], []


                    for mtd in self.Methods: 
                        with gr.Column(visible=mtd == self.Methods[0]) as show_temp:
                            otp_temp = gr.Model3D(label=f'模板牙齿模型 ({mtd})')
                            show_templates.append(show_temp)
                            otp_templates.append(otp_temp)
                        with gr.Column(visible=False) as show_slice:
                            otp_slice = gr.Gallery(
                                label=f"完整牙齿SDF切面 ({mtd})", show_label=True,
                                columns=[3], rows=[1], object_fit="fill", height='500'
                            )
                            show_slices.append(show_slice)
                            otp_slices.append(otp_slice)
                        with gr.Column(visible=False) as show_error:
                            otp_error = gr.Plot(
                                label=f"逐点误差图 ({mtd})", show_label=True
                            )
                            show_errors.append(show_error)
                            otp_errors.append(otp_error)
        
            ############ EVENTS ##############

            # logic for change
            def on_location_change(loc: str):
                locid = f'{loc[1:]}_Outside.obj'
                result = {}

                for mtd in self.Methods:
                    mtd_str = mtd.__str__()
                    temp_path = None
                    if locid in mtd.templates:
                        temp_path = mtd.templates[locid]
                    
                    result[otp_templates[reverse_index[mtd_str]]] = temp_path

                return result
            
            def on_show_off_change(methods: list, miscs: list, posts: list, loc):
                if len(methods) == 0:
                    gr.Warning('至少选择一个补全方法')
    
                _addon_total = len(miscs) > 1 or (len(miscs) == 1 and self.MISC[0] not in miscs)
                _addon_total = _addon_total or len(posts) > 0
                _show_temp = self.MISC[1] in miscs
                _show_slice = self.POST[0] in posts
                _show_error = self.POST[1] in posts

                result = {
                    show_total: gr.update(visible=_addon_total),
                    **on_location_change(loc)
                }
                for mtd in self.Methods:
                    mtd = mtd.__str__()
                    result[show_otps[reverse_index[mtd]]] = gr.update(visible=mtd in methods)
                    result[show_templates[reverse_index[mtd]]] = gr.update(visible=_show_temp and mtd in methods)
                    result[show_slices[reverse_index[mtd]]] = gr.update(visible=_show_slice and mtd in methods)
                    result[show_errors[reverse_index[mtd]]] = gr.update(visible=_show_error and mtd in methods)

                return result

            def on_submit_click(
                    mesh_path: str, mtds: list, loc: str, iter: int, resol: str, miscs: list,
                    sample_up: int, sample_mid: int, sample_down: int, smooth: int,
                    progress=gr.Progress(track_tqdm=True)
                ):
                inst_methods = [self.Methods[reverse_index[mtd]] for mtd in mtds]

                self._ctx = Context(mesh_path, inst_methods, loc, int(iter), self.RESOL_MAP[resol], self.MISC[0] in miscs)

                progress(0, desc='正在进行点云采样')
                self._ctx.preprocess([sample_down, sample_mid, sample_up])

                progress(0, desc='正在进行符号距离场重建')
                self._ctx.reconstruct_latent()

                progress(0, desc='正在提取零等值面')
                # extract at xxx_raw.obj
                self._ctx.extract_surface()

                progress(0, desc='正在后处理')
                meshes, slices, errors  = self._ctx.post_compute(smooth)
                '''
                meshes = {
                    'ToothDIT': 'path/to/obj',
                    'DIF-Net': None
                }
                slices = {
                    ...
                }
                errors = {
                    ...
                }
                '''
                result = {}
                for mtd in self.Methods:
                    mtd = mtd.__str__()
                    result[otp_meshes[reverse_index[mtd]]] = meshes.get(mtd, None)
                    result[otp_slices[reverse_index[mtd]]] = slices.get(mtd, None)
                    result[otp_errors[reverse_index[mtd]]] = errors.get(mtd, None)

                return result

            def on_upload_mesh(inp_mesh_path: str):
                if os.path.splitext(inp_mesh_path)[-1] != '.obj':
                    gr.Warning('非OBJ格式的3D模型暂不支持前端渲染，不影响重建过程')
                if os.path.isfile(inp_mesh_path):
                    return { btn_commit: gr.update(interactive=True) }
                
                gr.Warning('上传文件失败')
                return { btn_commit: gr.update(interactive=False) }

            def on_smooth_change(smooth: int, progress=gr.Progress(track_tqdm=True)):
                progress(0, desc='正在进行网格平滑处理')
                meshes, slices, errors = {}, {}, {}
                if self._ctx and self._ctx.finished:
                    meshes, slices, errors = self._ctx.post_compute(smooth)

                result = {}
                for mtd in self.Methods:
                    mtd = mtd.__str__()
                    result[otp_meshes[reverse_index[mtd]]] = meshes.get(mtd, None)
                    result[otp_slices[reverse_index[mtd]]] = slices.get(mtd, None)
                    result[otp_errors[reverse_index[mtd]]] = errors.get(mtd, None)

                return result

            ######### Listen Events ###########

            [
                inp_inst.change(
                    on_show_off_change,
                    inputs=[inp_methods, inp_miscs, inp_post, inp_loc],
                    outputs=[show_total, *show_otps, *show_templates, *show_slices, *show_errors, *otp_templates]
                )
                for inp_inst in [inp_miscs, inp_methods, inp_post] # 这三个控件都对应同一个事件
            ]
            # inp_miscs.change(
            #     on_show_off_change,
            #     inputs=[inp_methods, inp_miscs, inp_post, inp_loc],
            #     outputs=[show_total, *show_otps, *show_templates, *show_slices, *show_errors, *otp_templates]
            # )
            # inp_methods.change(
            #     on_show_off_change,
            #     inputs=[inp_methods, inp_miscs, inp_post, inp_loc],
            #     outputs=[show_total, *show_otps, *show_templates, *show_slices, *show_errors, *otp_templates]
            # )
            # inp_post.change(
            #     on_show_off_change,
            #     inputs=[inp_methods, inp_miscs, inp_post, inp_loc],
            #     outputs=[show_total, *show_otps, *show_templates, *show_slices, *show_errors, *otp_templates]
            # )
            inp_loc.change(
                on_location_change,
                inputs=[inp_loc],
                outputs=[*otp_templates]
            )
            inp_mesh.change(
                on_upload_mesh,
                inputs=[inp_mesh],
                outputs=[btn_commit]
            )
            inp_smooth.release(
                on_smooth_change,
                inputs=[inp_smooth],
                outputs=[*otp_meshes, *otp_slices, *otp_errors]
            )
            btn_commit.click(
                on_submit_click,
                inputs=[
                    inp_mesh, inp_methods, inp_loc,
                    inp_iter, inp_resol, inp_miscs,
                    inp_sample_up, inp_sample_mid, inp_sample_down,
                    inp_smooth
                ],
                outputs=[*otp_meshes, *otp_slices, *otp_errors]
            )


    def launch_public(self, host='0.0.0.0', port=8000):
        self.inst.queue().launch(share=False, server_name=host, server_port=port)
