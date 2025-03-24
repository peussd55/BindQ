from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from function import *
import json  # JSON 응답을 처리하기 위해 추가
import numpy as np
import sys
import ray

if "/home/eps/prj_envs/Gradio/ProteinMPNN/af_backprop" not in sys.path:
    sys.path.append("/home/eps/prj_envs/Gradio/ProteinMPNN/af_backprop")

from run_pipeline import save_uploaded_file, process_inputs, run_inference_ProteinMPNN
from run_pipeline import HOST_INPUT_DIR, HOST_RFDIFFUSION_OUTPUT_DIR
from ProteinMPNN.app import get_mpnn_ui

ray.shutdown()

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 제공 (예: static 폴더)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/rfdiffusion_output", StaticFiles(directory="rfdiffusion_output"), name="rfdiffusion_output")

# mol star 뷰어 cdn 경로
# <script src="https://cdn.jsdelivr.net/npm/molstar@latest/build/viewer/molstar.js"></script>
# <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@latest/build/viewer/molstar.css" />
# mol star 뷰어 정적파일 경로
# <script src="/static/molstar.js"></script>
# <link rel="stylesheet" type="text/css" href="/static/molstar.css" />
# sacnnet 뷰어 정적파일 경로
# <script src="/static/scannet/scannetmolstar.js"></script>
# <link rel="stylesheet" type="text/css" href="/static/scannet/molstar.css" />
# <script src="/static/scannet/utils.js"></script>
# <link rel="stylesheet" type="text/css" href="/static/scannet/style.css" />

# 사이드바 버튼글자 왼쪽정렬, 사이드바 색상변경
css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css');

.lg.secondary.svelte-1ixn6qd::before {
    color: #FFFFE0; /* 이제 .sidebar-icon 클래스를 가진 <i> 태그에만 색상 적용 */
    font-size: 1.2em;
}
.svelte-1ixn6qd {
    text-align: left !important;
    justify-content: flex-start !important;
}
.svelte-7n3dca {
    background-color: #000033 !important;    /* 오른쪽 화면 바탕색 변경 */

.svelte-y4v1h1 {
    background-color: #000033 !important;    /* 사이드바 바탕색 변경 */
}
input.svelte-173056l.svelte-173056l,  /* [🌟 중요 🌟] 입력창 선택자 (input 태그) */
textarea.svelte-173056l.svelte-173056l { /* [🌟 중요 🌟] 입력창 선택자 (textarea 태그) - 혹시 몰라서 추가 */
    background-color: #white !important; /* [🌟 핵심! 🌟] 입력창 바탕색을 #1c366b 로 변경! */
    color: black !important;             /* 글자색 흰색 유지 (가독성 확보) - 이미 설정되어 있을 듯, 혹시 몰라서 추가 */
    /* 필요에 따라 추가 스타일 (테두리, 폰트 등) */
}
.gradio-container .sidebar .markdown {
    color: white !important;
}
#tab0_content {
    background-color: #000033 !important;
    border: none !important;
    padding: 20px !important;
    box-shadow: none !important;
}

#tab0_content img {
    border: none !important;
}

/* 사이드바 내 모든 버튼에 적용되는 스타일 */
.gradio-container .sidebar button {
    /* 크기 조정 */
    width: 100%; /* 버튼 너비를 사이드바 너비에 맞춤 */
    padding: 8px 16px; /* 버튼 내부 여백 증가 (상하, 좌우) */
    /* margin-bottom: 2px; 버튼 사이 간격 */
    
    /* 테두리 둥글기 조정 */
    border-radius: 8px !important; /* 더 둥근 모서리 */
}

/* [[ button style 요소 설정하기]] */

/* my_button_style  */
#send-button {   /* elem_id="send-button"인 버튼 선택 */
    width: 100px !important; /* 가로 사이즈 변경 */
    border-radius: 10px !important; /* (선택 사항) 버튼 모서리 둥글게 */
    padding: 5px 10px !important; /* (선택 사항) 버튼 내부 여백 조정 padding (상하, 좌우)*/
    margin-left: auto !important; /* 가로 가운데 정렬 */
    margin-right: auto !important; /* 가로 가운데 정렬 */
    display: block !important; /* 블록 요소 설정 */
    font-size: 15px !important; /* 글자 크기 20px로 설정 */
    text-align: center !important; /* 텍스트 가운데 정렬 */
}
#main_button { /* elem_id="main_button"인 버튼 선택, Home화면 버튼 적용 */
    background-color: #003399 !important; /* 배경색 변경 */
    color: white !important; /* 글자색 변경 */
    width: 250px !important; /* 가로 사이즈 변경 */
    border-radius: 30px !important; /* (선택 사항) 버튼 모서리 둥글게 */
    padding: 10px 20px !important; /* (선택 사항) 버튼 내부 여백 조정 padding (상하, 좌우)*/
    margin-left: auto !important; /* 가로 가운데 정렬 */
    margin-right: auto !important; /* 가로 가운데 정렬 */
    display: block !important; /* 블록 요소 설정 */
    font-size: 18px !important; /* 글자 크기 20px로 설정 */
    text-align: center !important; /* 텍스트 가운데 정렬 */
}
#click_button { /* elem_id="click_button"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    background-color: #003399 !important; /* 배경색 변경 */
    color: white !important; /* 글자색 변경 */
    /* width: 250px !important;  가로 사이즈 변경 */
    border-radius: 10px !important; /* (선택 사항) 버튼 모서리 둥글게 */
    padding: 10px 20px !important; /* (선택 사항) 버튼 내부 여백 조정 padding (상하, 좌우)*/
    margin-left: auto !important; /* 가로 가운데 정렬 */
    margin-right: auto !important; /* 가로 가운데 정렬 */
    display: block !important; /* 블록 요소 설정 */
    font-size: 15px !important; /* 글자 크기 20px로 설정 */
    text-align: center !important; /* 텍스트 가운데 정렬 */
}
#click_button2 { /* elem_id="click_button2"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    background-color: #003399 !important; /* 배경색 변경 */
    color: white !important; /* 글자색 변경 */
    width: 153px !important;  가로 사이즈 변경
    border-radius: 10px !important; /* (선택 사항) 버튼 모서리 둥글게 */
    padding: 10px 20px !important; /* (선택 사항) 버튼 내부 여백 조정 padding (상하, 좌우)*/
    margin-left: auto !important; /* 가로 가운데 정렬 */
    margin-right: auto !important; /* 가로 가운데 정렬 */
    font-size: 15px !important; /* 글자 크기 20px로 설정 */
    text-align: center !important; /* 텍스트 가운데 정렬 */
}
#height_m { /* elem_id="height_m"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    height: 620px !important;
}
#height_m2 { /* elem_id="height_m"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    height: 50px !important;
}
#height { /* elem_id="height"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    height: 90px !important;
}
/* Chatbot 출력창 바탕색 변경 */
.wrapper.svelte-g3p8na {
    background-color: #ffffff !important; /* 후보: #020244, #081E40,#020244, */
    color: white !important;        /* 글자색  */

}
/* Chatbot 입력창 바탕색 변경 */
div.svelte-633qhp .block {
    background-color: #020244 !important; /* 원하는 배경색 */
    color: white !important; /* 텍스트 색상 흰색 유지 (가독성 위해) */

}
.svelte-i3tvor.float {
    display: none !important;
}
/* gr.Textbox 글자색을 흰색으로 변경 (span.svelte-1gfkn6j 선택자 사용 */
span.svelte-1gfkn6j { /* label 텍스트 요소 선택자: span.svelte-1gfkn6j */
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}
span.svelte-5ncdh7 { 
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}

.svelte-7ddecg h3, .svelte-7ddecg h2, .svelte-7ddecg h1, .svelte-7ddecg p { 
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}
.svelte-1tcem6n{ 
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}
.svelte-1tcem6n:hover{ 
    color: gray !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}

.svelte-ydeks8{ /* 에러 메시지 흰색고정 */
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}
.svelte-ydeks8 div{ /* 에러 메시지 흰색고정 */
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}
.svelte-ydeks8 p{ /* 에러 메시지 흰색고정 */
    color: white !important; /* label 글자색을 흰색으로 변경! */
    /* font-weight: bold !important; /* (선택 사항) label 폰트 굵게 - 필요하면 주석 해제 */
}

"""
icons = ["static/icons/Home_W.png", "static/icons/puzzle_W.png", "static/icons/line3_W.png", "static/icons/2_W.png", "static/icons/chart_W.png"]
file_path = ''  # 실제 CSV 파일 경로로 변경
sort_column = ''  # 정렬에 사용할 열 이름

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Sidebar():
            #home_logo = gr.Image(value="static/icons/logo_99.png", container=False, show_fullscreen_button=False, show_download_button=False, show_label=False, height=70)    
            logo_html = gr.HTML(f"""
                <img src="/static/icons/sidelogo.png" height="70" style="cursor: pointer;" 
                    onclick="window.location.reload();">
            """)

            tab_buttons = []
            for i, tab in enumerate(tabs):  # 기존 탭 버튼 생성 로직 재활용
                btn = gr.Button(
                    tab,
                    icon=icons[i],
                    variant="huggingface"
                    )
                tab_buttons.append(btn)

            # Add some spacing between tab buttons and chatbot
            gr.Markdown("<br><br>")
            gr.Markdown("<hr>")
            
            # Add chatbot section title
            gr.Markdown(value='### <span style="color:white"> 🗨️BindQ chat</span>'  )
            
            # Add the chatbot component
            chatbot = gr.Chatbot(height=250, show_copy_button=True, elem_classes=["gr-chatbot"]) # [🌟 변경 🌟] elem_classes 추가
            
            # Add message input and send button
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="메시지를 입력하세요...", 
                    label="", 
                    show_label=False
                )
                send_btn = gr.Button(value="메시지 보내기", variant="huggingface", elem_id="send-button")
            
            # Connect the send button and Enter key to the chat function
            send_btn.click(
                chat_response, 
                inputs=[msg, chatbot], 
                outputs=[msg, chatbot]
            )
            msg.submit(
                chat_response, 
                inputs=[msg, chatbot], 
                outputs=[msg, chatbot]
            )
        
        # 오른쪽 콘텐츠 영역
        with gr.Column(scale=8):
            # 각 탭 콘텐츠를 담을 컨테이너 생성
            tab_contents = []
            
            # 소개 탭 (기본 표시)
            with gr.Column(visible=True, elem_id="tab0_content") as tab0_content:
                gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[0]}</span>")
                gr.Image(value="static/icons/mainlogo.png", container=False, show_fullscreen_button=False, show_download_button=False, show_label=False)

                button = gr.Button("BindQ 알아보기", elem_id="main_button")
                image1 = gr.Image("static/1th_guide.png", visible=False, label="가이드0")
                visible_state = gr.State(False)
                button.click(
                    toggle_image_0,
                    inputs=[visible_state],
                    outputs=[image1, visible_state]
                )

            tab_contents.append(tab0_content)
            
            # Mol* Viewer 탭
            with gr.Column(visible=False) as tab1_content:
                # gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[1]}</span>")
                # with gr.TabItem("0. 타겟 단백질 구조예측"):
                #     input_box = gr.Textbox(label="PDB ID 또는 UniProt ID 입력", placeholder="4HHB 또는 P68871")
                #     molstar_output = gr.HTML()
                #     download_link = gr.HTML()
                    
                #     # 텍스트박스 값 변경 시 Mol* Viewer 업데이트
                #     input_box.submit(generate_molstar_html, inputs=input_box, outputs=[download_link])
                    
                #     # Flask 서버로 요청 보내기 버튼 추가
                #     send_button = gr.Button("단백질 결합 부위 예측하기", elem_id="click_button")
                #     flask_response_output = gr.Textbox(label="계산 결과 응답", interactive=False)
                    
                #     # 버튼 클릭 시 Flask 서버로 요청 보내기 동작 연결
                #     send_button.click(send_to_flask_server, inputs=input_box, outputs=[flask_response_output])
                #with gr.TabItem("구조기반 결합부위 예측"):
                gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[1]}</span>")

                with gr.Row():
                    with gr.Column():
                        input_box_3scannet = gr.Textbox(label="단백질 ID 입력", placeholder="ex) P01116, P68871 ...")
                        file_input = gr.File(label="Upload CSV File")
                        flag_rf = gr.Number(value=0, visible=False)
                        calc_pred_button = gr.Button("예측 수행", elem_id="click_button")
                        bsrp_output = gr.Textbox(label="DeepBSRPred 모델 수행 상태")

                    with gr.Column():
                        with gr.Column():
                            group_size = gr.Number(value=3, label="연속 시퀀스 갯수")
                            threshold_input = gr.Slider(0, 1, value=0.5, step=0.01, label="임계치")                            
                        with gr.Column():
                            scannet_weight = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.01, label="ScanNet 모델 가중치", interactive=True
                            )
                            bsrp_weight = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.01, label="DeepBSRPred 모델 가중치", interactive=True
                            )
                        scannet_weight.change(update_weights, inputs=scannet_weight, outputs=bsrp_weight)

                process_button = gr.Button("결과 조회", elem_id="click_button")
                with gr.Row():
                    with gr.Column(scale=5):
                        molstar_scannet_output = gr.HTML()
                    with gr.Column(scale=3):
                        merge_table = gr.Dataframe(
                            interactive=False,
                            visible=False,
                            max_height=900 
                        )  
                        
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 결합 예측 부위 리스트")
                        summary_table = gr.Dataframe(headers=["순위(Rank)", "컨티그(Contig)", "평균값(AVG)"], interactive=False)
                        summary_file = gr.File(visible=False)
                    with gr.Column():
                        gr.Markdown("### 상세 정보")
                        #details_table = gr.Dataframe(headers=["Chain", "Residue Index", "Sequence", "Binding site probability", "pred_prob", "combine_pred"], interactive=False)
                        details_table = gr.Dataframe(headers=["사슬", "잔기 번호", "시퀀스", "결합 부위 확률", "예측 확률", "예측 결합"], interactive=False)

                #calc_pred_button.click(fn=run_pred, inputs=[file_input, flag_rf], outputs=[bsrp_output])
                calc_pred_button.click(fn=click_pred_scannet, inputs=[input_box_3scannet, file_input, flag_rf], outputs=[bsrp_output])
                #process_button.click(fn=process_uploaded_file, inputs=[file_input, group_size, threshold_input, scannet_weight, bsrp_weight], outputs=[summary_table, summary_file, merge_table, molstar_scannet_output])
                process_button.click(fn=click_process_uploaded_file, inputs=[input_box_3scannet, file_input, group_size, threshold_input, scannet_weight, bsrp_weight], outputs=[summary_table, summary_file, merge_table, molstar_scannet_output])
            
                summary_table.select(fn=show_details, inputs=[summary_table], outputs=[details_table])

                # with gr.TabItem("결합가능 부위 예측"):
                #     input_box_3scannet1 = gr.Textbox(label="PDB ID 또는 UniProt ID 입력", placeholder="4HHB 또는 P68871")
                #     send_button = gr.Button("단백질 결합 부위 예측하기", elem_id="click_button")
                #     flask_response_output = gr.Textbox(label="결과", interactive=False, visible=True)
                #     flag_rf = gr.Number(value=0, visible=False)
                #     send_button.click(send_to_flask_server, inputs=[input_box_3scannet1, flag_rf], outputs=[flask_response_output])
                #     with gr.Row():
                #         with gr.Column(scale=8):
                #             molstar_scannet_output = gr.HTML()
                #             input_box_3scannet1.submit(generate_molstar_html_scannet, inputs=input_box_3scannet1, outputs=molstar_scannet_output)
                        
                #         with gr.Column(scale=3):
                #             data_table = gr.Dataframe(
                #                 value=load_and_sort_csv(file_path, sort_column),
                #                 interactive=True
                #             )
                #             input_box_3scannet1.submit(update_data_table, inputs=input_box_3scannet1, outputs=data_table)
            tab_contents.append(tab1_content)

            # RFdiffusion 탭
            with gr.Column(visible=False) as tab2_content:
                gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[2]}</span>")
                with gr.Row():
                    with gr.Column():
                        pdb_code_rf = gr.Textbox(label="단백질 ID 입력", placeholder="ex) P01116, P68871 ...")
                        # search_button = gr.Button("Download PDB")

                        # 파일 업로드 컴포넌트 (단일 파일)
                        pdb_file_rf = gr.File(label="PDB 파일 업로드", file_types=[".pdb", ".zip"], file_count="single")
                        flag_rf = gr.Number(value=0, visible=False)

                        # 저장 버튼
                        upload_button_rf = gr.Button("파일 업로드", elem_id="click_button")
                        # 저장 결과를 출력할 Textbox
                        save_output = gr.Textbox(label="저장 상태")
                        name_output = gr.Textbox(
                            label='파일명',
                            placeholder='샘플'
                        )
                        upload_button_rf.click(fn=handle_upload, inputs=[pdb_code_rf, pdb_file_rf, flag_rf], outputs=[save_output, name_output])

                    with gr.Column():
                        with gr.Row(elem_id="height"):
                            # RFdiffusion Flag
                            contigs = gr.Textbox(
                                label='컨티그(contigs)',
                                placeholder='예: 10-40/A163-181/10-40'
                            )
                        with gr.Row(elem_id="height"):
                            # RFdiffusion Flag
                            hotspot_res = gr.Textbox(
                                label='핫스팟 잔기(hotspot_res)',
                                placeholder='예: A59,A83,A91',
                                value=None
                            )

                        with gr.Row(elem_id="height"):
                            # RFdiffusion Flag
                            num_designs = gr.Dropdown(
                                choices=[str(i) for i in range(1, 11)],
                                label="생성 횟수",
                                value="5",
                                interactive=True
                            )

                        with gr.Row(elem_id="height"):
                            # with gr.Column():
                            # RFdiffusion Flag
                            iterations = gr.Slider(
                                minimum=25, maximum=200, value=50, step=1, label="Iterations", interactive=True, visible=False
                            )
                    
                with gr.Column():
                    # 출력창
                    #output = gr.Textbox(label="Output")
                    output_html = gr.HTML(label="3D Viewer")

                    # Inference 실행 버튼
                    run_button = gr.Button("생성시작", elem_id="click_button")

                    # 버튼 클릭 시 run_inference 함수 호출
                    #run_button.click(fn=run_inference, inputs=override_flags, outputs=output)
                    run_button.click(
                        fn=process_inputs,
                        inputs=[
                            name_output,
                            contigs,
                            hotspot_res,
                            iterations,
                            num_designs
                        ],
                        #outputs=output
                        outputs=output_html
                    )
                    #render_button = gr.Button("Render Structures", elem_id="click_button")
                    #render_button.click(render_pdbs_by_prefix, inputs=name_output, outputs=output_html)
            tab_contents.append(tab2_content)
            
            # ProteinMPNN 탭
            with gr.Column(visible=False) as tab3_content:
                    gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[3]}</span>")
                    get_mpnn_ui()
            tab_contents.append(tab3_content)
            
            # Prodigy 탭
            with gr.Column(visible=False) as tab4_content:
                    gr.Markdown(f"<span style='color: white; font-size: 20px;'>{tabs[4]}</span>")

                    # html = """
                    # <a href="https://cluspro.org/login.php" target="_blank">
                    #     <button style="padding: 10px; width: 300px; height: 100px; font-size: 36px; background-color: blue; color: white; border: none; cursor: pointer;">
                    #         ClusPro 바로가기
                    #     </button>
                    # </a>
                    # """
                    #link = gr.HTML(html, visible=False)

                    button = gr.Button("도킹 검증(단백질 - 단백질 상호작용 시뮬레이터)", elem_id="click_button")
                    button_cluspro = gr.Button("ClusPro 바로가기", visible=False, elem_id="click_button2")
                    image = gr.Image("static/5th_guide1.png", visible=False, label="가이드2", container=False, show_fullscreen_button=False, show_download_button=False, show_label=False)
                    visible_state = gr.State(False)
                    button.click(
                        toggle_image,
                        inputs=[visible_state],
                        outputs=[button_cluspro, image, visible_state]
                    )

                    button_cluspro.click(fn=None, inputs=None, outputs=None, js="() => { window.open('https://cluspro.org/login.php', '_blank') }")
                    file_input = gr.File(label="Docking 파일 업로드", file_types=[".pdb", ".zip"], file_count="single")
                    #interactor_1  = gr.Textbox(label="Interactor 1 *", placeholder="A or A,B")
                    #interactor_2  = gr.Textbox(label="Interactor 2 *", placeholder="A or A,B")
                    
                    css = """
                    .scroll-hide::-webkit-scrollbar {
                        display: none;
                    }
                    .scroll-hide {
                        -ms-overflow-style: none;
                        scrollbar-width: none;
                    }
                    """
                    head_nm = ['[파일명]', '예측된 결합 친화도 (ΔG) (kcal.mol-1)', '25.0˚C에서의 예측된 해리 상수 (Kd) (M)', '분자간 접촉 수', '전하-전하 접촉 수', '전하-극성 접촉 수', '전하-비극성 접촉 수', '극성-극성 접촉 수', '비극성-극성 접촉 수', '비극성-비극성 접촉 수', '비극성 NIS 잔기의 백분율', '전하를 띤 NIS 잔기의 백분율']
                    results_df = gr.Dataframe(label="분석 결과", headers=head_nm,  column_widths=["200px", "200px", "150px"], wrap=True, interactive=False, elem_classes=["scroll-hide"])
                    save_button = gr.Button("단백질 - 단백질 복합체의 결합 친화도 예측", elem_id="click_button")
                    output_html = gr.HTML(label="3D Viewer")

                    save_button.click(fn=prodigy_process, inputs=[file_input], outputs=[results_df, output_html])
            tab_contents.append(tab4_content)
    
    # 각 버튼에 클릭 이벤트 연결
    for i, btn in enumerate(tab_buttons):
        btn.click(
            fn=lambda idx=i: show_tab_content(idx), 
            outputs=tab_contents
        )


# Gradio 앱을 FastAPI에 마운트
app = gr.mount_gradio_app(app, demo, path="/gradio")

# FastAPI 실행 (uvicorn 필요)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True, reload_includes=["app.py"])