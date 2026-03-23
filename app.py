import os
import streamlit as st
import pandas as pd
from PIL import Image
import torch
import config
from config import logger
from inference import load_model, predict_calib_center, predict_exp_keypoints
from utils.delta_x_calculator import calculate_delta_x
from utils.gray_visualizer import plot_gray_profile

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="AI+毛细波衍射实验系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 自定义CSS样式 =====================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ===================== 标题 =====================
st.title("🔬 AI+毛细波衍射条纹间距测量系统")
st.markdown("---")


# ===================== 加载模型（缓存，避免重复加载） =====================
@st.cache_resource(show_spinner="正在加载AI模型，请稍候...")
def get_inference_model():
    """加载并缓存模型"""
    device = torch.device("cpu")  # 学生本地用CPU
    try:
        model = load_model(config.BEST_MODEL_PATH, device)
        return model, device
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        st.stop()


# ===================== 会话状态初始化 =====================
if "calib_center" not in st.session_state:
    st.session_state.calib_center = None  # 存储calib中央条纹坐标
if "calib_img" not in st.session_state:
    st.session_state.calib_img = None  # 存储calib图像
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []  # 存储实验历史

# ===================== 侧边栏：第一步 上传calib图 =====================
with st.sidebar:
    st.header("📌 第一步：上传标定图")
    st.write("（无振针激励的图片，用于标定中央条纹位置）")

    # 上传calib图
    calib_file = st.file_uploader(
        "选择标定图（支持JPG/PNG）",
        type=["jpg", "png", "jpeg"],
        key="calib_uploader"
    )

    if calib_file is not None:
        # 显示calib图
        st.session_state.calib_img = Image.open(calib_file)
        st.image(
            st.session_state.calib_img,
            caption="上传的标定图",
            use_container_width=True
        )

        # 标定按钮
        if st.button("开始标定中央条纹", key="calib_button"):
            with st.spinner("🤖 AI正在标定中央条纹位置..."):
                try:
                    model, device = get_inference_model()
                    st.session_state.calib_center = predict_calib_center(
                        model,
                        st.session_state.calib_img,
                        device
                    )
                    st.success(f"✅ 标定成功！")
                    st.info(
                        f"中央条纹坐标：({st.session_state.calib_center[0]:.1f}, {st.session_state.calib_center[1]:.1f})")
                except Exception as e:
                    st.error(f"标定失败：{e}")

    # 显示当前标定状态
    st.markdown("---")
    if st.session_state.calib_center is not None:
        st.success("✅ 标定状态：已完成")
    else:
        st.warning("⚠️ 标定状态：未完成")

# ===================== 主页面：第二步 上传exp图做实验 =====================
st.header("🧪 第二步：上传实验图")
st.write("（有振针激励的图片，可多次上传进行多组实验）")

# 检查是否已标定
if st.session_state.calib_center is None:
    st.warning("⚠️ 请先在左侧上传标定图并完成标定！")
else:
    # 上传exp图
    exp_file = st.file_uploader(
        "选择实验图（支持JPG/PNG）",
        type=["jpg", "png", "jpeg"],
        key="exp_uploader"
    )

    if exp_file is not None:
        # 显示exp图
        exp_img = Image.open(exp_file)
        st.image(
            exp_img,
            caption="上传的实验图",
            use_container_width=True
        )

        # 计算按钮
        if st.button("计算条纹间距并显示灰度图", key="exp_button"):
            with st.spinner("🤖 AI正在处理实验数据..."):
                try:
                    model, device = get_inference_model()

                    # 预测exp图的关键点
                    exp_first_order, scale_0mm, scale_10mm = predict_exp_keypoints(
                        model,
                        exp_img,
                        device
                    )

                    # 计算delta_x
                    delta_x, pixel_per_mm = calculate_delta_x(
                        st.session_state.calib_center,
                        exp_first_order,
                        scale_0mm,
                        scale_10mm
                    )

                    # 绘制灰度图
                    gray_fig = plot_gray_profile(
                        exp_img,
                        st.session_state.calib_center,
                        exp_first_order,
                        scale_0mm,
                        scale_10mm
                    )

                    # ===================== 显示结果 =====================
                    st.markdown("---")
                    st.subheader("📊 实验结果")

                    # 用三列显示指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="条纹间距 (delta_x)",
                            value=f"{delta_x:.2f} mm"
                        )
                    with col2:
                        st.metric(
                            label="像素-毫米比例",
                            value=f"{pixel_per_mm:.1f} 像素/mm"
                        )
                    with col3:
                        st.metric(
                            label="实验序号",
                            value=f"{len(st.session_state.experiment_history) + 1}"
                        )

                    # 显示灰度图
                    st.subheader("📈 条纹亮度分析")
                    st.pyplot(gray_fig)

                    # ===================== 保存实验历史 =====================
                    st.session_state.experiment_history.append({
                        "实验序号": len(st.session_state.experiment_history) + 1,
                        "delta_x (mm)": round(delta_x, 2),
                        "像素-毫米比例": round(pixel_per_mm, 1),
                        "中央条纹坐标": f"({st.session_state.calib_center[0]:.1f}, {st.session_state.calib_center[1]:.1f})",
                        "一级条纹坐标": f"({exp_first_order[0]:.1f}, {exp_first_order[1]:.1f})"
                    })

                except Exception as e:
                    st.error(f"处理失败：{e}")

    # ===================== 显示实验历史 =====================
    st.markdown("---")
    st.subheader("📋 实验历史记录")

    if len(st.session_state.experiment_history) > 0:
        # 转换为DataFrame
        history_df = pd.DataFrame(st.session_state.experiment_history)
        st.dataframe(history_df, use_container_width=True)

        # 导出CSV按钮
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 导出实验记录为CSV",
            data=csv,
            file_name="毛细波衍射实验记录.csv",
            mime="text/csv"
        )

        # 清空历史按钮
        if st.button("🗑️ 清空实验历史"):
            st.session_state.experiment_history = []
            st.experimental_rerun()
    else:
        st.info("暂无实验记录，请上传实验图开始实验。")