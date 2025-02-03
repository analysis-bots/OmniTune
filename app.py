import time

import streamlit as st
import pandas as pd
import os

from agents.constraint_parser import ConstraintParser
from agents.function_generation import ImplementationAgent
from agents.langgraph_oracle import run_task
from functionality.task import AgnosticTask
from tools.sql_engine import SQLEngineAdvanced
from tools import constants
from streamlit.components.v1 import html
from tools.utils import parse_where_predicates, parse_where_clause, extract_where_clause, highlight_predicate_differences_getter

DEBUG = False

DATASETS = {
    "Law Students": "law_students.csv",
    "Astronauts": "astronauts_500kb.csv",
    "Students": "students.csv",
    "Insurance": "medical_insurance.csv",
    "Texas Tribune": "texas_tribune.csv",
    "Healthcare": "healthcare.csv",
}


def generate_sql_query(conditions, query_suffix=""):
    prefix = f"SELECT * FROM df "
    for c in conditions:
        if c['operator'] == "IN":
            try:
                c['value'] = str(tuple(c['value'])).replace(",)", ")")
            except AttributeError:
                ...
    where_clause = " AND ".join([f"\"{c['attribute']}\" {c['operator']} {c['value']}" for c in conditions])
    query = f"{prefix}{where_clause}\n{query_suffix}"
    return query


def load_dataset(selected_dataset):
    dataset_path = os.path.join('datasets', DATASETS[selected_dataset])
    return pd.read_csv(dataset_path)

def change_button_shape(button_index=0, margin_top=0, left=0):
    html(f"""
    <script>
        const buttons = window.parent.document.querySelectorAll(".stButton > button");
        const lastButton = buttons[{button_index}];
        lastButton.style.border = 'none';                   
        lastButton.style.background = 'none';
        lastButton.style.padding = '0';
        lastButton.style.color = 'blue';
        lastButton.style.textDecoration = 'underline';
        lastButton.style.marginTop = '{margin_top}px';
        lastButton.style.position = 'relative';
        lastButton.style.left = '{left}px';
        
        // Add hover effect using mouseover and mouseout events
        lastButton.addEventListener('mouseover', () => {{
            lastButton.style.textDecoration = 'underline'; // Remove text decoration
            lastButton.style.color = 'red'; // Change text color
        }});

        lastButton.addEventListener('mouseout', () => {{
            lastButton.style.textDecoration = 'underline'; // Restore text decoration
            lastButton.style.color = 'blue'; // Restore text color
        }});
    </script>
    """, height=0)


def switch_tab_js(tab_index):
    return f"""
    <script>
        var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
        var tabs = tabGroup.getElementsByTagName("button");
        tabs[{tab_index}].click();
    </script>
    """


def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: Black; font-size: 62px;'>OmniTune</h1>",
                unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: Black; font-size: 62px;'>OmniTune"
    #             "<span style=' vertical-align: super	;font-size: 21px;'>©</span></h1>", unsafe_allow_html=True)
    left_half, ___, right_half = st.columns([9.5, 0.5, 11])
    if 'got_input' not in st.session_state:
        st.session_state.got_input = False

    if 'invalid_query' not in st.session_state:
        st.session_state.invalid_query = False

    if 'show_constraints' not in st.session_state:
        st.session_state.show_constraints = False

    if 'show_refinement' not in st.session_state:
        st.session_state.show_refinement = False

    if 'show_output' not in st.session_state:
        st.session_state.show_output = False

    if "toggle_constraints" not in st.session_state:
        st.session_state.toggle_constraints = False  # False means "Off", True means "On"

    if "toggle_refinement" not in st.session_state:
        st.session_state.toggle_refinement = False  # False means "Off", True means "On"

    if "spinner_state" not in st.session_state:
        st.session_state.spinner_state = "Refining Query..."

    if "specifics_provided" not in st.session_state:
        st.session_state.specifics_provided = False

    if "specifics" not in st.session_state:
        st.session_state.specifics = None

    if "toggle_specifics" not in st.session_state:
        st.session_state.toggle_specifics = False

    if "user_uploaded" not in st.session_state:
        st.session_state.user_uploaded = False

    with left_half:
        st.markdown("<h4 style='margin-bottom: 10px;'>OmniTune Refinement Problem Wizard</h4>", unsafe_allow_html=True)
        # st.markdown("<h4 style='margin-bottom: 10px;'>Enter new query refinement problem:</h4>", unsafe_allow_html=True)

        # Add a divider with custom spacing
        st.markdown(
            "<hr style='margin-top: -25px; margin-bottom: 10px;'>",
            unsafe_allow_html=True
        )

    with right_half:
        tabs = ["Databases"]

        if st.session_state["show_constraints"] and len(tabs) == 1:
            tabs.append("Constraints Function")

        if st.session_state["show_refinement"] and len(tabs) == 2:
            tabs.append("Refinement Distance")

        if st.session_state["show_output"] and len(tabs) == 3:
            tabs.append("Output")
            tabs.append("Log & Previous Queries")

        # Create the tabs
        tab_objects = st.tabs(tabs)

        with tab_objects[0]:
            tabs2 = ["Input Database"]
            if st.session_state.got_input:
                tabs2.append("Original Query Result")

            def switch_tab_js2(tab_index):
                return f"""
                <script>
                    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[1];
                    var tabs = tabGroup.getElementsByTagName("button");
                    tabs[{tab_index}].click();
                </script>
                """

            input_tab_objects = st.tabs(tabs2)

    if 'params' not in st.session_state:
        st.session_state.params = {}

    if 'show_dataframe' not in st.session_state:
        st.session_state.show_dataframe = False

    if 'show_output_dataframe' not in st.session_state:
        st.session_state.show_output_dataframe = False

    if 'constraints_objective_getter' not in st.session_state:
        st.session_state.constraints_objective_getter = None

    if 'refinement_objective_getter' not in st.session_state:
        st.session_state.refinement_objective_getter = None

    ######################## Part 1: Query Input ########################
    def query_input_section():
        # Add custom instructions

        with upload_col2:
            # File uploader
            uploaded_file = st.file_uploader("", type=["csv"], key="file_uploader", label_visibility="collapsed")
            if uploaded_file is not None:
                # Read the uploaded CSV file
                try:
                    hide_file_name_css = """
                        <style>
                        [data-testid="stFileUploader"] > div{
                            visibility: hidden;
                            height: 0;
                        }
                        </style>
                        """
                    st.markdown(hide_file_name_css, unsafe_allow_html=True)
                    st.session_state.dataset = pd.read_csv(uploaded_file)
                    st.markdown("""<div style="font-size: 12px; color: green; position: relative; top: -30px; margin-bottom: 0px;">
                                        Database uploaded successfully.
                                    </div>""", unsafe_allow_html=True)
                    st.session_state.user_uploaded = True
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            # else:
            #     st.session_state.user_uploaded = True
        # Load dataset and get attributes
        with right_half:
            with tab_objects[0]:
                with input_tab_objects[0]:
                    st.dataframe(st.session_state.dataset, height=350, width=1200)

        db_name = st.session_state.selected_database.replace(" ", "_").lower()
        # Add a big input text box
        st.markdown("""
                <div style="font-size: 16px; color: black; position: relative; top: -40px; margin-top: -30px;">
                    Query to refine: 
                </div>
                """,
                    unsafe_allow_html=True)
        user_input = st.text_area(
            "",
            height=50,
            value=f"SELECT * FROM {db_name}",
        )
        if st.session_state.invalid_query:
            st.markdown(
                "<div style='color: red; font-size: 14px; margin-top: -10px;'>Invalid query. Please try again.</div>",
                unsafe_allow_html=True)
        st.markdown("""<style>
            .stTextArea {
                position: relative !important;
                top: -80px !important;
            }
            </style>""", unsafe_allow_html=True)
        return db_name, user_input

    with left_half:
        upload_col1, upload_col2 = st.columns([6, 1])
        with upload_col1:
            st.markdown("""
            <style>
                [data-testid='stFileUploader'] {
                    width: max-content;
                    margin-top: 19px;
                    position: relative;
                    left: -10px;
                }
                [data-testid='stFileUploader'] section {
                    padding: 0;
                    float: left;
                }
                [data-testid='stFileUploader'] section > input + div {
                    display: none;
                }
                [data-testid='stFileUploader'] section + div {
                    float: right;
                    padding-top: 0;
                }

            </style>
                    """, unsafe_allow_html=True)
            st.markdown("""
                    <div style="font-size: 16px; color: black; position: relative; top: -30px; margin-bottom: 0px;">
                        Connect to database or upload CSV files: 
                    </div>
                    """,
                        unsafe_allow_html=True)
            st.session_state.selected_database = st.selectbox("", list(DATASETS.keys()),
                                                              index=0)
            st.markdown("""<style>
                .stSelectbox {
                    display: inline !important;
                    position: relative !important;
                    top: -50px !important;
                }
                </style>""", unsafe_allow_html=True)
            st.session_state.dataset = load_dataset(st.session_state.selected_database)

        if not st.session_state.got_input:
            # Shared logic for query input
            db_name, original_query = query_input_section()
            st.session_state.params["db_name"] = db_name
            cont_button_1_placeholder = st.empty()
            st.markdown("""<style>
                .stButton {
                    position: relative !important;
                    top: -95px !important;
                }
                </style>""", unsafe_allow_html=True)
            if cont_button_1_placeholder.button("Continue", key="cont_query", use_container_width=True):
                dataset = st.session_state.dataset
                dfs = {db_name: dataset}
                st.session_state.input_dataset = dataset
                st.session_state.params["dfs"] = dfs
                if original_query:
                    st.session_state.params["query"] = original_query
                    sql_engine = SQLEngineAdvanced(dfs, keep_index=False)

                    # TODO - add exception handling for invalid queries
                    st.session_state.original_output_dataset, is_exception = sql_engine.execute(original_query)
                    if is_exception or not "WHERE" in original_query:
                        st.session_state.params["query"] = None
                        st.session_state.invalid_query = True
                        st.rerun()
                    st.session_state.invalid_query = False
                    st.session_state.got_input = True
                    st.session_state.show_constraints = True
                    st.rerun()
    if st.session_state.got_input:
        with right_half:
            with tab_objects[0]:
                html(switch_tab_js2(1))
                with input_tab_objects[0]:
                    st.dataframe(st.session_state.input_dataset, height=350, width=1200)
                with input_tab_objects[1]:
                    st.dataframe(st.session_state.original_output_dataset, height=350, width=1200)
                original_query = st.session_state.params["query"]
                st.session_state.params["agent"] = ImplementationAgent(original_query, st.session_state.params["dfs"])
                st.session_state.highlight_predicate_differences = highlight_predicate_differences_getter(original_query)
        with left_half:
            with upload_col2:
                # File uploader
                hide_file_name_css = """
                    <style>
                    [data-testid="stFileUploader"] > div{
                        visibility: hidden;
                        height: 0;
                    }
                    </style>
                    """
                st.markdown(hide_file_name_css, unsafe_allow_html=True)
                st.file_uploader("", type=["csv"], key="file_uploader")
                if st.session_state.user_uploaded:
                    st.markdown("""<div style="font-size: 12px; color: green; position: relative; top: -18px; margin-bottom: 0px;">
                                        Database uploaded successfully.
                                    </div>""", unsafe_allow_html=True)
                st.markdown("""
                <style>
                    [data-testid='stFileUploader'] {
                        width: max-content;
                        margin-top: -24px;
                    }
                    </style>
                        """, unsafe_allow_html=True)

            try:
                cont_button_1_placeholder.empty()
            except UnboundLocalError:
                pass
            if "agent" in st.session_state.params:
                st.markdown("""
                        <div style="font-size: 16px; color: black; position: relative; top: -40px; margin-top: -30px;">
                            Query to refine: 
                        </div>
                        """,
                            unsafe_allow_html=True)
                st.text_area("", height=50, value=st.session_state.params["query"], key="query_to_refine2")
                st.markdown("""<style>
                    .stTextArea {
                        margin-top: -80px !important;
                    }
                    </style>""", unsafe_allow_html=True)

    ######################## Part 2: Constraints ########################
    if st.session_state.show_constraints:
        with left_half:
            if "agent" in st.session_state.params:
                # Define the toggle button
                if "user_constraints_code" not in st.session_state and not st.session_state.toggle_constraints:
                    cons_col1, cons_col2, cons_col3 = st.columns([19, 3, 9])
                    with cons_col1:
                        st.markdown("""
                                <div style="font-size: 16px; color: black; display:inline;">
                                    Provide a description of the constraints in a natural language, or
                                </div>
                                """,
                                    unsafe_allow_html=True)
                    with cons_col2:
                        if st.button("click here", key="toggle_constraints_button"):
                            st.session_state.toggle_constraints = not st.session_state.toggle_constraints
                            st.rerun()
                        change_button_shape(button_index=0, left=-55, margin_top=-20)
                    with cons_col3:
                        st.markdown("""
                                <div style="font-size: 16px; display:inline; position: relative; left: -66px;">
                                     to provide a python function.
                                </div>
                                """,
                                    unsafe_allow_html=True)
                    function_placeholder = st.empty()
            if st.session_state.toggle_constraints:
                with right_half:
                    html(switch_tab_js(1), height=0)
                    with tab_objects[1]:
                        tab_col1, tab_col2, tab_col3 = st.columns([24, 4, 18])

                        # Inject custom CSS
                        with tab_col1:
                            st.markdown(
                                """
                                <div style="font-size: 16px; color: black;">
                                    Insert your Code below. Your code should include the threshold logic, or
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        ORIG_VALUE = """def get_constraints_satisfaction_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
                            # Outer function to assess constraints satisfaction
                            def constraints_satisfaction_objective(refined_query: str) -> float:
                                # TODO - Your code goes here
                                ...
                                return 1.0
                            return constraints_satisfaction_objective
                        """
                        with tab_col2:
                            if st.button("click here", key="toggle_constraints_button2",
                                         use_container_width=True):
                                st.session_state.toggle_constraints = not st.session_state.toggle_constraints
                                st.rerun()
                            change_button_shape(left=-27, margin_top=-20)
                        with tab_col3:
                            st.markdown(
                                """
                                <div style="font-size: 16px; color: black; position: relative; left: -40px;">
                                    to provide a natural language description.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        constraints_objective_getter = st.text_area("", value=ORIG_VALUE, height=300)
                        # Toggle the state
                        if st.button("Accept Code", key="accept_code"):
                            st.session_state.constraints_objective_getter = constraints_objective_getter

            elif "user_constraints_code" not in st.session_state:
                with right_half:
                    html(switch_tab_js(0), height=0)
                with left_half:
                    function_description = function_placeholder.text_area("", height=50)
                    generate_button_placeholder = st.empty()
                    if generate_button_placeholder.button("Generate Constraints Satisfaction Objective", key="generate_constraints"):
                        st.session_state.params["refinement_objective_str"] = function_description
                        with left_half:
                            generate_button_placeholder.empty()
                        with right_half:
                            html(switch_tab_js(1))
                            with tab_objects[1]:
                                spinner_placeholder = st.empty()
                                with spinner_placeholder, st.spinner("Generating Constraint Satisfaction Deviation Objective..."):
                                    st.session_state.params["constraints_str"] = function_description
                                    agent = st.session_state.params["agent"]
                                    constraints_objective_getter = agent.generate_constraints_objective(function_description)
                                    st.session_state.constraints_objective_getter = constraints_objective_getter
                                    st.rerun()
    if st.session_state.constraints_objective_getter:
        if "user_constraints_code" not in st.session_state:
            with right_half:
                if st.session_state.show_constraints:
                    if "constraints_objective_getter_updated" not in st.session_state:
                        st.session_state.constraints_objective_getter_updated = st.session_state.constraints_objective_getter
                    html(switch_tab_js(1), height=0)
                    with tab_objects[1]:
                        constraints_logic()
            try:
                with left_half:
                    st.markdown(
                        "<div style='font-size: 15px; color: green'>Constraints function was Successfully generated via GPT-4o-mini.</div>",
                        unsafe_allow_html=True)
                    generate_button_placeholder.empty()
                    cont_button_2_placeholder = st.empty()
                    if cont_button_2_placeholder.button("Continue", key="cont_constraints", use_container_width=True):
                        st.session_state.user_constraints_code = st.session_state.constraints_objective_getter
                        st.session_state.show_refinement = True
                        generate_button_placeholder.empty()
                        st.rerun()

            except UnboundLocalError:
                st.rerun()

        else:
            # Show the final code in the right half in the Constraints tab
            with right_half:
                html(switch_tab_js(1), height=0)
                with tab_objects[1]:
                    with st.container(height=600):
                        st.code(st.session_state.constraints_objective_getter, language="python",
                                line_numbers=True,
                                wrap_lines=True)
            with left_half:
                try:
                    generate_button_placeholder.empty()
                except UnboundLocalError:
                    pass
                cons_col1, cons_col2, cons_col3 = st.columns([35, 6, 18])

                with cons_col1:
                    st.markdown("""
                            <div style="font-size: 16px; color: black; display:inline; position: relative; top: -20px;">
                                Provide the a description of the constraints in a natural language, or
                            </div>
                            """,
                                unsafe_allow_html=True)
                with cons_col2:
                    st.markdown("""
                            <div style="font-size: 16px; color: blue; display:inline; text-decoration: underline; position: relative; top: -20px; left: -16px;">
                                 click here
                            </div>
                            """,
                                unsafe_allow_html=True)
                with cons_col3:
                    st.markdown("""
                            <div style="font-size: 16px; color: black; display:inline; position: relative; top: -20px; left: -32px;">
                                to provide a python function.
                            </div>
                            """,
                                unsafe_allow_html=True)

                st.markdown("")
                st.markdown("")
                st.text_area("", value=st.session_state.params["constraints_str"], height=50)
                st.markdown(
                    "<div style='font-size: 15px; color: green; margin-top: -10px'>Constraints function was Successfully generated via GPT-4o-mini.</div>",
                    unsafe_allow_html=True)

    ######################## Part 3: Refinement Objective ########################

    if st.session_state.show_refinement and st.session_state.refinement_objective_getter is None:

        with left_half:
            try:
                cont_button_1_placeholder.empty()
            except UnboundLocalError:
                pass
            # Define the toggle button
            if st.session_state.toggle_refinement:
                with right_half:
                    html(switch_tab_js(2))
                    with tab_objects[2]:
                        refinement_objective_getter = st.text_area(
                            "Insert your Refinement Objective Code here:", height=300)
                        if st.button("Accept Code", key="accept_refinement_code"):
                            st.session_state.refinement_objective_getter = refinement_objective_getter

            elif "user_distance" not in st.session_state:
                with left_half:
                    st.markdown(
                        "<div style='font-size: 16px; color: black; margin-top: 10px;'>Select refinement distance method:</div>",
                        unsafe_allow_html=True)
                    st.session_state.refinement_type = st.radio(
                        "",
                        ('Query based', 'Result based', 'Define custom'),
                        horizontal=True)
                    html("""<script>
                        const radio = window.parent.document.querySelectorAll(".stRadio");
                        const lastRadio = radio[radio.length - 1];  
                        lastRadio.style.position = 'relative';
                        lastRadio.style.top = '-25px';
                    </script>""", height=0)

                    if st.session_state.refinement_type == 'Query based':
                        st.session_state.toggle_specifics = False
                        refinement_objective_getter = constants.REFINEMENT_QUERY_DISTANCE
                        st.session_state.params["refinement_objective_str"] = constants.REFINEMENT_QUERY_DESCRIPTION
                    elif st.session_state.refinement_type == "Result based":
                        st.session_state.toggle_specifics = False
                        refinement_objective_getter = constants.REFINEMENT_RESULT_DISTANCE
                        st.session_state.params["refinement_objective_str"] = constants.REFINEMENT_RESULT_DESCRIPTION
                    else:
                        st.session_state.toggle_specifics = True
                    if st.session_state.toggle_specifics:
                        with left_half:
                            with st.expander("Provide Specifics for the Refinement Objective"):
                                st.markdown(" ")
                                st.markdown(" ")
                                st.markdown(" ")
                                st.markdown(" ")
                                st.session_state.specifics = st.text_area(
                                    "", height=50)

                            html("""<script>
                                const expander = window.parent.document.querySelectorAll(".stExpander")[0];
                                expander.style.position = 'relative';
                                expander.style.top = '-60px';
                            </script>""", height=0)

                    generate_button_2_placeholder = st.empty()
                    if generate_button_2_placeholder.button("Generate Refinement Distance Objective", key="generate_refinement"):
                        with right_half:
                            html(switch_tab_js(2))
                            with tab_objects[2]:
                                if st.session_state.toggle_specifics:
                                    spinner_placeholder = st.empty()
                                    with spinner_placeholder, st.spinner("Generating Refinement Distance Objective..."):
                                        st.session_state.params["refinement_objective_str"] = constants.REFINEMENT_QUERY_DESCRIPTION
                                        agent = st.session_state.params["agent"]
                                        refinement_objective_getter = agent.generate_refinement_distance_objective(st.session_state.specifics)
                                st.session_state.refinement_objective_getter = refinement_objective_getter
                                st.rerun()
                    px_val = -100 if st.session_state.toggle_specifics else -70
                    html(f"""<script>
                        const buttons = window.parent.document.querySelectorAll(".stButton > button");
                        const lastButton = buttons[buttons.length - 1];
                        lastButton.style.position = 'relative';
                        lastButton.style.top = '{px_val}px';
                    </script>""", height=0)

    if st.session_state.refinement_objective_getter:
        try:
            generate_button_2_placeholder.empty()
        except UnboundLocalError:
            pass

        options = [i for i in range(0, 101, 5)]

        def f(x):
            if x == 0:
                return ""
            elif x == 100:
                return ""
            return f"{x}%"

        if "user_distance" not in st.session_state:
            with right_half:
                html(switch_tab_js(2))
                with tab_objects[2]:
                    refinement_logic()
        else:
            with right_half:
                with tab_objects[2]:
                    with st.container(height=600):
                        st.code(st.session_state.refinement_objective_getter, language="python",
                                line_numbers=True,
                                wrap_lines=True)
        with left_half:
            if st.session_state.refinement_type == 'Query based':
                radio_index = 0
            elif st.session_state.refinement_type == 'Result based':
                radio_index = 1
            else:
                radio_index = 2

            st.markdown(
                "<div style='font-size: 16px; color: black; margin-top: 5px;'>Select refinement distance method:</div>",
                unsafe_allow_html=True)
            st.radio(
                "",
                ('Query based', 'Result based', 'Define custom'),
                horizontal=True,
                index=radio_index)
            html("""<script>
                const radio = window.parent.document.querySelectorAll(".stRadio");
                const lastRadio = radio[radio.length - 1];  
                lastRadio.style.position = 'relative';
                lastRadio.style.top = '-15px';
            </script>""", height=0)


            if st.session_state.toggle_specifics:
                with st.expander("Provide Specifics for the Refinement Objective", expanded=False):
                    st.markdown(" ")
                    st.markdown(" ")
                    st.markdown(" ")
                    st.session_state.specifics = st.text_area(
                        "", value=st.session_state.specifics, height=50)
            st.markdown(
                "<div style='font-size: 16px; position: relative; top: -50px;'>Constraint Satisfaction Threshold:</div>",
                unsafe_allow_html=True)

            epsilon = st.select_slider(
                "Constraint Satisfaction Threshold:",
                options=options,
                format_func=f,
                value=50,
                label_visibility='collapsed'
            )

            st.markdown("""<style>
                .stSlider {
                    margin-top: -40px !important;
                }
                </style>""", unsafe_allow_html=True)

            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: -40px;">
                    <span>Min. Refinement Changes</span>
                    <span>Max. Constraints Satisfaction</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state.params["epsilon"] = 1 - float(epsilon / 100)
            if st.button("Start Refinement", key="cont_refinement2", use_container_width=True):
                st.session_state.user_distance = st.session_state.refinement_objective_getter
                st.session_state.show_output = True
                st.rerun()

            html("""<script>
                const buttons = window.parent.document.querySelectorAll(".stButton > button");
                const lastButton = buttons[buttons.length - 2];
                lastButton.style.position = 'relative';
                lastButton.style.top = '-2px';
            </script>""", height=0)


    ######################## Part 4: Output ########################
    if st.session_state.show_output:
        with right_half:
            html(switch_tab_js(3))
            name = st.session_state.params["db_name"]
            d_in = st.session_state.params["dfs"]
            original_query = st.session_state.params["query"]
            constraints_str = st.session_state.params["constraints_str"]
            refinement_objective_str = st.session_state.params["refinement_objective_str"]

            exec(st.session_state.user_constraints_code)
            constraints_satisfaction_objective = eval("get_constraints_satisfaction_objective(d_in, original_query)")
            exec(st.session_state.user_distance)
            refinement_objective = eval("get_refinement_distance_objective(d_in, original_query)")

            epsilon = st.session_state.params["epsilon"]

            st.session_state.task = AgnosticTask(name, d_in, original_query, constraints_str,
                                                 refinement_objective_str,
                                                 constraints_satisfaction_objective, refinement_objective, epsilon)

            if "output_df" not in st.session_state:
                with tab_objects[3]:
                    best_query, constraint_feedback, chat_writer, constraint_score, \
                        refinement_score = run_task(st.session_state.task, output_tab=tab_objects[3],
                                                    chat_tab=tab_objects[4], is_agnostic=True)
                st.session_state.chat_writer = chat_writer
                st.session_state.best_query = best_query
                st.session_state.constraint_score = constraint_score
                st.session_state.refinement_score = refinement_score
                st.session_state.constraint_feedback = constraint_feedback
                st.session_state.highlighted_diff = st.session_state.highlight_predicate_differences(st.session_state.best_query)
                sql_engine = SQLEngineAdvanced(d_in)
                st.session_state.orig_df, _ = sql_engine.execute(st.session_state.params["query"])
                st.session_state.output_df, _ = sql_engine.execute(st.session_state.best_query)
                st.session_state.refinements_count = chat_writer.refinements_counter - 1

                only_in_orig = st.session_state.orig_df[~st.session_state.orig_df.index.isin(st.session_state.output_df.index)]
                concat_no_dupes = pd.concat([st.session_state.output_df, only_in_orig])

                if len(only_in_orig) == 0:
                    st.session_state.is_diff = False
                    # Rounding to two decimal places
                    old_df = st.session_state.orig_df.round(2)
                    new_df = st.session_state.output_df.round(2)

                    # Create a combined DataFrame with formatted values
                    combined_df = old_df.copy()

                    for col in combined_df.columns:
                        combined_df[col] = [
                            f"<span style='color:red;text-decoration:line-through'>{old}</span> → <span style='color:green'>{new}</span>"
                            if old != new else f"<span>{new}</span>"
                            for old, new in zip(old_df[col], new_df[col])
                        ]

                    # Convert the combined DataFrame to an HTML table
                    st.session_state.styled_df = combined_df.to_html(escape=False, index=False)
                else:
                    st.session_state.is_diff = True
                    # Style function to highlight specific cells
                    def highlight_row(row):
                        if row.name not in st.session_state.orig_df.index:
                            return ["background-color: lightgreen"] * len(row)
                        elif row.name not in st.session_state.output_df.index:
                            return ["background-color: lightcoral"] * len(row)
                        else:
                            return [""] * len(row)
                    # Apply the style to the DataFrame
                    st.session_state.styled_df = concat_no_dupes.style.apply(highlight_row, axis=1)

                st.session_state.chat_writer.reset_output()
            with tab_objects[3]:
                output_logic()


def output_logic():
    st.markdown(
        f"<div style='font-size: 20px;'><b>Final Refined Query (after  {st.session_state.refinements_count} attempts):</b></div>",
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family: monospace; white-space: pre-wrap;">{st.session_state.highlighted_diff}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(" ")
    st.markdown(
        f"<div style='position: relative; top: -10px; font-size: 17px;'><b> Constraints Deviation Score: </b>{st.session_state.constraint_score:.2f}</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='position: relative; top: -10px; font-size: 17px;'><b> Refinement Distance Score: </b>{st.session_state.refinement_score:.2f}</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 20px; position: relative; top: -7px;'><b>Constraints Score explanation:</b></div>",
        unsafe_allow_html=True)
    st.markdown("\n".join(st.session_state.constraint_feedback))
    st.markdown(
        "<div style='position: relative; top: -10px; font-size: 20px;'><b>Refined Query Results:</b></div>",
        unsafe_allow_html=True)
    if not st.session_state.is_diff:
        # Add custom CSS to style the table like a st.dataframe
        st.markdown(
            """
            <style>
            .dataframe-container {
                width: 100%;
                max-height: 350px; /* Adjust max height to control scroll area */
                overflow: auto;
                position: relative;
                top: -40px;
                border: none;
                border-radius: 8px;
                margin-top: 20px;
                border-bottom: 1px solid #ddd;
                border-right: none;
                resize: vertical;
            }
            table {
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
            }
            th {
                background-color: #f4f4f4;
                color: #333;
                font-weight: normal;
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            tr:last-child td {
                border-bottom: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Wrap the table in a scrollable container
        st.markdown(
            f"""
            <div class="dataframe-container">
                {st.session_state.styled_df}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div style="position: relative; top: -35px;"><font color="green"><b>*Green values</b>: only in the refined query result</font> '
                    '<font color="red"><b>*Red values</b>: only in the original query result</font></div>', unsafe_allow_html=True)
    else:
        st.dataframe(st.session_state.styled_df, height=320, width=1200)
        st.markdown('<div style="position: relative; top: -20px;"><font color="green"><b>*Green values</b>: only in the refined query result</font> '
                    '<font color="red"><b>*Red values</b>: only in the original query result</font></div>', unsafe_allow_html=True)


def refinement_logic():
    placeholder = st.empty()
    if "user_distance" in st.session_state:
        with placeholder.container(height=700):
            st.code(st.session_state.refinement_objective_getter, language="python",
                    line_numbers=True, wrap_lines=True)
    else:
        if st.session_state.refinement_objective_getter:
            with placeholder.container(height=700):
                st.code(st.session_state.refinement_objective_getter, language="python",
                        line_numbers=True, wrap_lines=True)

            if "refinement_objective_getter_updated" not in st.session_state:
                st.session_state.refinement_objective_getter_updated = st.session_state.refinement_objective_getter

            if "ref_edit_mode" not in st.session_state:
                st.session_state.ref_edit_mode = False

            placeholder_button = st.empty()
            if st.session_state.ref_edit_mode:
                with placeholder.container(height=620):
                    st.session_state.refinement_objective_getter_updated = st.text_area(
                        "Below is the generated Refinement Objective Code. "
                        "To edit, press the Edit button. When ready, press Accept:",
                        value=st.session_state.refinement_objective_getter,
                        height=620)
                if placeholder_button.button("Accept", key="accept_refinement", use_container_width=True):
                    st.session_state.ref_edit_mode = False

            if not st.session_state.ref_edit_mode:
                with placeholder.container(height=620):
                    st.code(st.session_state.refinement_objective_getter_updated, language="python",
                            line_numbers=True, wrap_lines=True)
                st.session_state.refinement_objective_getter = st.session_state.refinement_objective_getter_updated
                if placeholder_button.button("Edit", key="edit_refinement", use_container_width=True):
                    st.session_state.ref_edit_mode = True
                    with placeholder.container(height=620):
                        st.session_state.refinement_objective_getter_updated = st.text_area(
                            "Below is the generated Refinement Objective Code. "
                            "To edit, press the Edit button. When ready, press Accept:",
                            value=st.session_state.refinement_objective_getter,
                            height=620)
                    if placeholder_button.button("Accept", key="accept_refinement", use_container_width=True):
                        st.session_state.ref_edit_mode = False
                        with placeholder.container(height=620):
                            placeholder.code(st.session_state.refinement_objective_getter_updated,
                                             language="python",
                                             line_numbers=True, wrap_lines=True)
                        st.session_state.refinement_objective_getter = st.session_state.refinement_objective_getter_updated


def constraints_logic():
    placeholder = st.empty()
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False

    if st.session_state.edit_mode:
        with placeholder.container(height=600):
            st.session_state.constraints_objective_getter_updated = st.text_area(
                "Below is the generated Constraint Satisfaction Objective Code. To edit, press the Edit button."
                " When ready, press Accept:",
                value=st.session_state.constraints_objective_getter,
                height=600
            )
            placeholder_button = st.empty()
            if placeholder_button.button("Accept Code", key="accept_constraints", use_container_width=True):
                st.session_state.edit_mode = False

    if not st.session_state.edit_mode:
        with placeholder.container(height=600):
            st.code(st.session_state.constraints_objective_getter_updated, language="python", line_numbers=True,
                    wrap_lines=True)
        placeholder_button = st.empty()
        st.session_state.constraints_objective_getter = st.session_state.constraints_objective_getter_updated
        if placeholder_button.button("Edit Code", key="edit_constraints", use_container_width=True):
            st.session_state.edit_mode = True
            with placeholder.container(height=600):
                st.session_state.constraints_objective_getter_updated = st.text_area(
                    "Below is the generated Constraint Satisfaction Objective Code. To edit, press the Edit button."
                    " When ready, press Accept:",
                    value=st.session_state.constraints_objective_getter,
                    height=600
                )
            if placeholder_button.button("Accept Code", key="accept_constraints", use_container_width=True):
                st.session_state.edit_mode = False
                placeholder.code(st.session_state.constraints_objective_getter_updated, language="python",
                                 line_numbers=True,
                                 wrap_lines=True)
                st.session_state.constraints_objective_getter = st.session_state.constraints_objective_getter_updated
        # html("""
        # <script>
        #     const buttons = window.parent.document.querySelectorAll(".stButton > button");
        #     buttons[buttons.length - 1].style.position = "relative";
        #     buttons[buttons.length - 1].style.top = "0px";
        # </script>
        # """)

if __name__ == "__main__":
    main()
