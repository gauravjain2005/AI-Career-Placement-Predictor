import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Indian_Student_Placement_Dataset_2025.csv"


# =========================================================
# LOAD MODEL & DATASET
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model load nahi ho paya: {e}")
    st.stop()


try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Dataset load nahi ho paya: {e}")
    st.stop()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Career Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🎓 AI Career Placement Predictor")

st.caption(
    "AI/ML based student placement prediction and career preparation system."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 About Project")

    st.write(
        "This application uses Machine Learning to estimate "
        "student placement probability and provide career "
        "preparation suggestions."
    )

    st.divider()

    st.subheader("🛠 Technologies")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🤖 Scikit-learn")
    st.write("🌲 Random Forest")
    st.write("🎨 Streamlit")

    st.divider()

    st.subheader("📂 Dataset")

    st.write(f"Students: {len(df):,}")

    if "placed" in df.columns:
        placement_rate = df["placed"].mean() * 100
        st.write(f"Placement Rate: {placement_rate:.2f}%")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔮 Prediction",
    "📊 Analytics",
    "👨‍🎓 Student Explorer"
])


# =========================================================
# TAB 1 — DASHBOARD
# =========================================================

with tab1:

    st.header("🏠 Placement Dashboard")

    st.write(
        "Overview of the student placement dataset."
    )

    st.divider()

    # --------------------------------------
    # METRICS
    # --------------------------------------

    total_students = len(df)

    placed_students = (
        df["placed"] == 1
    ).sum()

    not_placed_students = (
        df["placed"] == 0
    ).sum()

    placement_rate = (
        placed_students / total_students
    ) * 100

    average_cgpa = df["cgpa"].mean()
    average_coding = df["coding_skills"].mean()
    average_aptitude = df["aptitude_score"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👨‍🎓 Total Students",
            f"{total_students:,}"
        )

    with col2:

        st.metric(
            "✅ Placed",
            f"{placed_students:,}"
        )

    with col3:

        st.metric(
            "❌ Not Placed",
            f"{not_placed_students:,}"
        )

    with col4:

        st.metric(
            "📈 Placement Rate",
            f"{placement_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------
    # AVERAGE PERFORMANCE
    # --------------------------------------

    st.subheader("📚 Average Student Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📊 Average CGPA",
            f"{average_cgpa:.2f}"
        )

    with col2:

        st.metric(
            "💻 Average Coding",
            f"{average_coding:.2f}/10"
        )

    with col3:

        st.metric(
            "🧠 Average Aptitude",
            f"{average_aptitude:.2f}/100"
        )

    st.divider()

    # --------------------------------------
    # PLACEMENT CHART
    # --------------------------------------

    st.subheader("📊 Placement Distribution")

    placement_chart = pd.DataFrame({
        "Status": [
            "Placed",
            "Not Placed"
        ],
        "Students": [
            placed_students,
            not_placed_students
        ]
    })

    st.bar_chart(
        placement_chart.set_index("Status")
    )

    # --------------------------------------
    # DATASET INSIGHT
    # --------------------------------------

    st.divider()

    st.subheader("💡 Dataset Insights")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:

        st.info(
            f"📚 Average CGPA: **{average_cgpa:.2f}**"
        )

    with insight_col2:

        st.info(
            f"💻 Average Coding: **{average_coding:.2f}/10**"
        )

    with insight_col3:

        st.info(
            f"🧠 Average Aptitude: **{average_aptitude:.2f}/100**"
        )


# =========================================================
# TAB 2 — PREDICTION
# =========================================================

with tab2:

    st.header("🔮 Student Placement Prediction")

    st.write(
        "Enter student details to estimate placement probability."
    )

    st.divider()

    # --------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------

    st.subheader("👤 Student Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=40,
            value=21,
            step=1
        )

    with col3:

        degree_options = sorted(
            df["degree"].dropna().unique().tolist()
        )

        degree = st.selectbox(
            "Degree",
            degree_options
        )

    with col4:

        branch_options = sorted(
            df["branch"].dropna().unique().tolist()
        )

        branch = st.selectbox(
            "Branch",
            branch_options
        )

    # --------------------------------------
    # ACADEMIC
    # --------------------------------------

    st.subheader("📚 Academic Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    with col2:

        backlogs = st.number_input(
            "Backlogs",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with col3:

        aptitude_score = st.number_input(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1
        )

    # --------------------------------------
    # SKILLS
    # --------------------------------------

    st.subheader("💻 Skills & Experience")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        coding_skills = st.number_input(
            "Coding Skills",
            min_value=0,
            max_value=10,
            value=6,
            step=1
        )

    with col2:

        communication_skills = st.number_input(
            "Communication",
            min_value=0,
            max_value=10,
            value=7,
            step=1
        )

    with col3:

        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    with col4:

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    with col5:

        projects = st.number_input(
            "Projects",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    st.divider()

    # --------------------------------------
    # PREDICT
    # --------------------------------------

    if st.button(
        "🚀 Predict Placement",
        use_container_width=True,
        type="primary"
    ):

        input_data = pd.DataFrame([{

            "gender": gender,
            "age": age,
            "degree": degree,
            "branch": branch,
            "cgpa": cgpa,
            "backlogs": backlogs,
            "internships": internships,
            "certifications": certifications,
            "coding_skills": coding_skills,
            "communication_skills": communication_skills,
            "aptitude_score": aptitude_score,
            "projects": projects

        }])

        try:

            prediction = model.predict(input_data)

            probability = model.predict_proba(
                input_data
            )[0][1]

        except Exception as e:

            st.error(
                f"Prediction ke time error aaya: {e}"
            )

            st.stop()

        # ----------------------------------
        # RESULT
        # ----------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction[0] == 1:

                st.success(
                    "🎉 Student is likely to be PLACED!"
                )

            else:

                st.error(
                    "❌ Student may need more preparation."
                )

        with result_col2:

            st.metric(
                "Placement Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(
            float(probability)
        )

        # ----------------------------------
        # PROFILE
        # ----------------------------------

        st.divider()

        st.subheader("📈 Student Profile Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📚 CGPA",
                f"{cgpa:.1f}/10"
            )

        with col2:

            st.metric(
                "💻 Coding",
                f"{coding_skills}/10"
            )

        with col3:

            st.metric(
                "🧠 Aptitude",
                f"{aptitude_score}/100"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🗣️ Communication",
                f"{communication_skills}/10"
            )

        with col2:

            st.metric(
                "🏢 Internships",
                internships
            )

        with col3:

            st.metric(
                "🚀 Projects",
                projects
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📜 Certifications",
                certifications
            )

        with col2:

            st.metric(
                "📋 Backlogs",
                backlogs
            )

        with col3:

            st.metric(
                "🎓 Degree",
                degree
            )

        # ----------------------------------
        # IMPORTANT MODEL FACTORS
        # ----------------------------------

        st.divider()

        st.subheader("🔑 Important Model Factors")

        factor1, factor2, factor3 = st.columns(3)

        with factor1:

            st.info(
                "💻 Coding Skills\n\n"
                "**37.78% feature importance**"
            )

        with factor2:

            st.info(
                "🧠 Aptitude Score\n\n"
                "**32.06% feature importance**"
            )

        with factor3:

            st.info(
                "📚 CGPA\n\n"
                "**22.18% feature importance**"
            )

        st.caption(
            "Feature importance values are based on the trained "
            "Random Forest model and should not be interpreted "
            "as causal effects."
        )

        # ----------------------------------
        # SUGGESTIONS
        # ----------------------------------

        st.subheader("💡 Improvement Suggestions")

        suggestions = []

        if coding_skills < 7:

            suggestions.append(
                "💻 Improve coding skills through DSA and regular practice."
            )

        if aptitude_score < 75:

            suggestions.append(
                "🧠 Practice quantitative aptitude and logical reasoning."
            )

        if cgpa < 7.5:

            suggestions.append(
                "📚 Focus on semester subjects and improve CGPA."
            )

        if internships == 0:

            suggestions.append(
                "🏢 Try to complete at least one internship."
            )

        if projects < 2:

            suggestions.append(
                "🚀 Build more practical projects."
            )

        if communication_skills < 6:

            suggestions.append(
                "🗣️ Practice communication and mock interviews."
            )

        if len(suggestions) == 0:

            st.success(
                "🔥 Keep improving consistently!"
            )

        else:

            for suggestion in suggestions:

                st.write(
                    "•",
                    suggestion
                )

        # ----------------------------------
        # CAREER PLAN
        # ----------------------------------

        st.divider()

        st.subheader(
            "🎯 Personalized Career Preparation Plan"
        )

        recommendations = []

        if coding_skills < 7:

            recommendations.append({
                "Area": "💻 Coding",
                "Current Level": f"{coding_skills}/10",
                "Recommendation":
                    "Practice DSA, Python and problem solving.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "💻 Coding",
                "Current Level": f"{coding_skills}/10",
                "Recommendation":
                    "Continue DSA and advanced problems.",
                "Priority": "Maintain"
            })

        if aptitude_score < 75:

            recommendations.append({
                "Area": "🧠 Aptitude",
                "Current Level": f"{aptitude_score}/100",
                "Recommendation":
                    "Practice aptitude and reasoning.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "🧠 Aptitude",
                "Current Level": f"{aptitude_score}/100",
                "Recommendation":
                    "Practice timed aptitude tests.",
                "Priority": "Maintain"
            })

        if cgpa < 7.5:

            recommendations.append({
                "Area": "📚 Academics",
                "Current Level": f"{cgpa}/10",
                "Recommendation":
                    "Focus on semester subjects and CGPA.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "📚 Academics",
                "Current Level": f"{cgpa}/10",
                "Recommendation":
                    "Maintain academic performance.",
                "Priority": "Maintain"
            })

        if projects < 2:

            recommendations.append({
                "Area": "🚀 Projects",
                "Current Level": str(projects),
                "Recommendation":
                    "Build at least 2 practical projects.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "🚀 Projects",
                "Current Level": str(projects),
                "Recommendation":
                    "Build advanced AI/ML projects.",
                "Priority": "Maintain"
            })

        if internships == 0:

            recommendations.append({
                "Area": "🏢 Internship",
                "Current Level": "0",
                "Recommendation":
                    "Apply for AI/ML or software internships.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "🏢 Internship",
                "Current Level": str(internships),
                "Recommendation":
                    "Gain more practical experience.",
                "Priority": "Maintain"
            })

        if communication_skills < 6:

            recommendations.append({
                "Area": "🗣️ Communication",
                "Current Level":
                    f"{communication_skills}/10",
                "Recommendation":
                    "Practice speaking and mock interviews.",
                "Priority": "High"
            })

        else:

            recommendations.append({
                "Area": "🗣️ Communication",
                "Current Level":
                    f"{communication_skills}/10",
                "Recommendation":
                    "Continue interview practice.",
                "Priority": "Maintain"
            })

        recommendation_df = pd.DataFrame(
            recommendations
        )

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------
        # 30 DAY PLAN
        # ----------------------------------

        st.subheader(
            "📅 Suggested 30-Day Preparation Plan"
        )

        st.write(
            "**Week 1:** Python + DSA basics + daily aptitude practice"
        )

        st.write(
            "**Week 2:** Coding problems + AI/ML project + communication"
        )

        st.write(
            "**Week 3:** Advanced DSA + resume improvement + mock tests"
        )

        st.write(
            "**Week 4:** Mock interviews + project explanation + placement preparation"
        )


# =========================================================
# TAB 3 — ANALYTICS
# =========================================================

with tab3:

    st.header("📊 Detailed Placement Analytics")

    st.write(
        "Explore placement patterns in the dataset."
    )

    st.divider()

    # --------------------------------------
    # CGPA ANALYSIS
    # --------------------------------------

    st.subheader("📚 CGPA vs Placement")

    cgpa_analysis = (
        df.groupby("placed")["cgpa"]
        .mean()
        .rename({
            0: "Not Placed",
            1: "Placed"
        })
    )

    st.bar_chart(
        cgpa_analysis
    )

    # --------------------------------------
    # CODING ANALYSIS
    # --------------------------------------

    st.subheader("💻 Coding Skills vs Placement")

    coding_analysis = (
        df.groupby("placed")["coding_skills"]
        .mean()
        .rename({
            0: "Not Placed",
            1: "Placed"
        })
    )

    st.bar_chart(
        coding_analysis
    )

    # --------------------------------------
    # APTITUDE ANALYSIS
    # --------------------------------------

    st.subheader("🧠 Aptitude vs Placement")

    aptitude_analysis = (
        df.groupby("placed")["aptitude_score"]
        .mean()
        .rename({
            0: "Not Placed",
            1: "Placed"
        })
    )

    st.bar_chart(
        aptitude_analysis
    )

    # --------------------------------------
    # BRANCH ANALYSIS
    # --------------------------------------

    st.subheader("🎓 Branch-wise Placement")

    branch_analysis = (
        df.groupby("branch")["placed"]
        .mean() * 100
    )

    branch_analysis = branch_analysis.sort_values(
        ascending=False
    )

    st.bar_chart(
        branch_analysis
    )

    # --------------------------------------
    # DEGREE ANALYSIS
    # --------------------------------------

    st.subheader("🎓 Degree-wise Placement")

    degree_analysis = (
        df.groupby("degree")["placed"]
        .mean() * 100
    )

    degree_analysis = degree_analysis.sort_values(
        ascending=False
    )

    st.bar_chart(
        degree_analysis
    )

    # --------------------------------------
    # APTITUDE RANGE ANALYSIS
    # --------------------------------------

    st.divider()

    st.subheader("🧠 Placement Rate by Aptitude Range")

    aptitude_bins = [0, 59, 69, 79, 89, 100]
    aptitude_labels = [
        "0–59",
        "60–69",
        "70–79",
        "80–89",
        "90–100"
    ]

    aptitude_range = pd.cut(
        df["aptitude_score"],
        bins=aptitude_bins,
        labels=aptitude_labels,
        include_lowest=True
    )

    aptitude_rate = (
        df.groupby(
            aptitude_range,
            observed=False
        )["placed"]
        .mean()
        .mul(100)
    )

    st.bar_chart(
        aptitude_rate
    )

    # --------------------------------------
    # CGPA RANGE ANALYSIS
    # --------------------------------------

    st.subheader("📚 Placement Rate by CGPA Range")

    cgpa_bins = [
        0,
        5.9,
        6.4,
        6.9,
        7.9,
        8.9,
        10
    ]

    cgpa_labels = [
        "0–5.9",
        "6.0–6.4",
        "6.5–6.9",
        "7.0–7.9",
        "8.0–8.9",
        "9.0–10"
    ]

    cgpa_range = pd.cut(
        df["cgpa"],
        bins=cgpa_bins,
        labels=cgpa_labels,
        include_lowest=True
    )

    cgpa_rate = (
        df.groupby(
            cgpa_range,
            observed=False
        )["placed"]
        .mean()
        .mul(100)
    )

    st.bar_chart(
        cgpa_rate
    )


# =========================================================
# TAB 4 — STUDENT EXPLORER
# =========================================================

with tab4:

    st.header("👨‍🎓 Student Explorer")

    st.write(
        "Search and filter students from the placement dataset."
    )

    st.divider()

    # --------------------------------------
    # FILTERS
    # --------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        placement_filter = st.selectbox(
            "Placement Status",
            [
                "All",
                "Placed",
                "Not Placed"
            ]
        )

    with col2:

        branch_filter = st.selectbox(
            "Branch",
            ["All"] + sorted(
                df["branch"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col3:

        degree_filter = st.selectbox(
            "Degree",
            ["All"] + sorted(
                df["degree"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    # --------------------------------------
    # SEARCH STUDENT ID
    # --------------------------------------

    student_search = st.text_input(
        "🔍 Search Student ID",
        placeholder="Enter Student ID"
    )

    filtered_df = df.copy()

    # --------------------------------------
    # PLACEMENT FILTER
    # --------------------------------------

    if placement_filter == "Placed":

        filtered_df = filtered_df[
            filtered_df["placed"] == 1
        ]

    elif placement_filter == "Not Placed":

        filtered_df = filtered_df[
            filtered_df["placed"] == 0
        ]

    # --------------------------------------
    # BRANCH FILTER
    # --------------------------------------

    if branch_filter != "All":

        filtered_df = filtered_df[
            filtered_df["branch"] == branch_filter
        ]

    # --------------------------------------
    # DEGREE FILTER
    # --------------------------------------

    if degree_filter != "All":

        filtered_df = filtered_df[
            filtered_df["degree"] == degree_filter
        ]

    # --------------------------------------
    # STUDENT ID SEARCH
    # --------------------------------------

    if student_search:

        filtered_df = filtered_df[
            filtered_df["student_id"]
            .astype(str)
            .str.contains(
                student_search,
                case=False,
                na=False
            )
        ]

    # --------------------------------------
    # RESULT COUNT
    # --------------------------------------

    st.metric(
        "👨‍🎓 Students Found",
        len(filtered_df)
    )

    # --------------------------------------
    # DISPLAY DATA
    # --------------------------------------

    display_columns = [
        "student_id",
        "gender",
        "age",
        "degree",
        "branch",
        "cgpa",
        "backlogs",
        "internships",
        "certifications",
        "coding_skills",
        "communication_skills",
        "aptitude_score",
        "projects",
        "placed"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    if len(filtered_df) > 0:

        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Koi student nahi mila. Filters ya Student ID check karo."
        )

    # --------------------------------------
    # DOWNLOAD FILTERED DATA
    # --------------------------------------

    csv_data = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Filtered Student Data",
        data=csv_data,
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ Predictions are estimates based on the trained dataset "
    "and are not a guarantee of placement."
)

st.caption(
    "AI Career Placement Predictor • Random Forest ML Model"
)