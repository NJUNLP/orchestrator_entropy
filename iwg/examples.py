"""
Complete case studies from the IWG paper.

Includes:
  1. "The White Ribbon" case — the running example from Section 3 / Appendix.
  2. "Miss Chris Audio Request" case — financial professional MAS workflow.

These demonstrate the full IWG pipeline end-to-end.
"""

import json
import sys
from typing import Optional

from .models import (
    AgentCapability,
    BenchmarkInstance,
    Checkpoint,
    CheckpointType,
    EnvironmentInfo,
    ExecutorDef,
    MASConfig,
    SeedData,
    TaskMark,
)
from .scout_agent import ScoutAgent
from .wrapper_agent import WrapperAgent
from .pipeline import IWGPipeline


# ===========================================================================
# "The White Ribbon" — default MAS configuration
# ===========================================================================

WHITE_RIBBON_MAS = MASConfig(
    description="Multi-agent system for multimedia information retrieval and GUI operations",
    executors=[
        ExecutorDef(
            name="VisionAgent",
            capabilities=[AgentCapability.VISION],
            description="Analyzes images, posters, and visual content to identify entities",
            tools=["image_recognition", "ocr", "object_detection"],
        ),
        ExecutorDef(
            name="EntityRetriever",
            capabilities=[AgentCapability.ENTITY_RETRIEVAL],
            description="Retrieves structured information about entities (people, films, dates)",
            tools=["knowledge_base_query", "wiki_lookup", "entity_linking"],
        ),
        ExecutorDef(
            name="GUIOperator",
            capabilities=[AgentCapability.GUI_OPERATION],
            description="Performs GUI operations like adding items to playlists, saving to dashboards",
            tools=["app_operation", "api_call", "form_submission"],
        ),
    ],
    max_steps=20,
)

# ===========================================================================
# "The White Ribbon" — seed data
# ===========================================================================

WHITE_RIBBON_SEED = SeedData(
    query="What year was the director of The White Ribbon born?",
    answer="March 23, 1942",
    domain="multimedia_qa",
    metadata={"film": "The White Ribbon", "director": "Michael Haneke", "birth_year": "1942"},
)

# ===========================================================================
# "The White Ribbon" — pre-constructed Task Marks (Scout output)
# ===========================================================================

WHITE_RIBBON_TASK_MARKS = [
    TaskMark(
        id="M_0",
        description="Identify the film from the movie poster using visual recognition",
        assigned_agent="VisionAgent",
        required_capability=AgentCapability.VISION,
        dependencies=[],
        checkpoint_hint="The White Ribbon",
        is_extension=False,
    ),
    TaskMark(
        id="M_1",
        description="Retrieve director information for the identified film",
        assigned_agent="EntityRetriever",
        required_capability=AgentCapability.ENTITY_RETRIEVAL,
        dependencies=["M_0"],
        checkpoint_hint="Michael Haneke",
        is_extension=False,
    ),
    TaskMark(
        id="M_2",
        description="Retrieve birth date of director Michael Haneke",
        assigned_agent="EntityRetriever",
        required_capability=AgentCapability.ENTITY_RETRIEVAL,
        dependencies=["M_1"],
        checkpoint_hint="March 23, 1942",
        is_extension=False,
    ),
    TaskMark(
        id="M_3",
        description="[Extended] Add the film to user's playlist via GUI operation",
        assigned_agent="GUIOperator",
        required_capability=AgentCapability.GUI_OPERATION,
        dependencies=["M_0"],
        checkpoint_hint="API(POST /playlist) -> status=200",
        is_extension=True,
    ),
]

# ===========================================================================
# "The White Ribbon" — pre-constructed Environment Info (Wrapper output)
# ===========================================================================

WHITE_RIBBON_ENVIRONMENTS = [
    EnvironmentInfo(
        id="EI_0",
        task_mark_id="M_0",
        step_index=0,
        agent_name="VisionAgent",
        tool_prompt="Observe the movie poster and identify the film title.",
        tool_output=(
            "The poster shows a black-and-white image of a rural village with "
            "children walking in a line. The title reads 'The White Ribbon' "
            "(original: 'Das weiße Band') in German with English subtitling. "
            "The film appears to be a period drama set in pre-WWI Germany. "
            "The cinematography suggests an art-house production."
        ),
        checkpoint=Checkpoint(
            id="CP_0",
            task_mark_id="M_0",
            checkpoint_type=CheckpointType.EXACT_MATCH,
            expected_value="The White Ribbon",
            verification_prompt="What is the film title identified from the poster?",
            step_index=0,
        ),
    ),
    EnvironmentInfo(
        id="EI_1",
        task_mark_id="M_1",
        step_index=1,
        agent_name="EntityRetriever",
        tool_prompt="Who directed the film 'The White Ribbon'?",
        tool_output=(
            "Entity: The White Ribbon (Das weiße Band)\n"
            "Type: Film\n"
            "Release Year: 2009\n"
            "Director: Michael Haneke\n"
            "Genre: Drama, Mystery\n"
            "Country: Germany, Austria, France, Italy\n"
            "Awards: Palme d'Or (2009), Golden Globe for Best Foreign Language Film\n"
            "Synopsis: A film by Michael Haneke set in rural Germany just before WWI, "
            "exploring themes of authority, violence, and the origins of fascism."
        ),
        checkpoint=Checkpoint(
            id="CP_1",
            task_mark_id="M_1",
            checkpoint_type=CheckpointType.EXACT_MATCH,
            expected_value="Michael Haneke",
            verification_prompt="Who directed The White Ribbon?",
            step_index=1,
        ),
    ),
    EnvironmentInfo(
        id="EI_2",
        task_mark_id="M_2",
        step_index=2,
        agent_name="EntityRetriever",
        tool_prompt="When was director Michael Haneke born?",
        tool_output=(
            "Entity: Michael Haneke\n"
            "Type: Person\n"
            "Profession: Film Director, Screenwriter\n"
            "Birth Date: March 23, 1942\n"
            "Birth Place: Munich, Germany\n"
            "Nationality: Austrian\n"
            "Notable Works: The White Ribbon (2009), Amour (2012), "
            "Caché (2005), Funny Games (1997)\n"
            "Awards: Two Palme d'Or awards, Academy Award for Best Foreign Language Film"
        ),
        checkpoint=Checkpoint(
            id="CP_2",
            task_mark_id="M_2",
            checkpoint_type=CheckpointType.EXACT_MATCH,
            expected_value="March 23, 1942",
            verification_prompt="When was Michael Haneke born?",
            step_index=2,
        ),
    ),
    EnvironmentInfo(
        id="EI_3",
        task_mark_id="M_3",
        step_index=3,
        agent_name="GUIOperator",
        tool_prompt="Add the film 'The White Ribbon' to the user's 'German Cinema' playlist.",
        tool_output=(
            "GUI Operation Result:\n"
            "Action: Add to Playlist\n"
            "Target: 'The White Ribbon' → Playlist 'German Cinema'\n"
            "Status: 200 OK\n"
            "Response: {{\"playlist_id\": \"pl_german_cinema\", "
            "\"added_item\": \"film_tt1345836\", \"position\": 12, "
            "\"playlist_count\": 12}}"
        ),
        checkpoint=Checkpoint(
            id="CP_3",
            task_mark_id="M_3",
            checkpoint_type=CheckpointType.API_VERIFY,
            expected_value="API(GET /playlist/pl_german_cinema) -> film_tt1345836",
            verification_prompt="Verify via API that The White Ribbon was added to the German Cinema playlist",
            step_index=3,
        ),
    ),
]

# ===========================================================================
# "The White Ribbon" — exception scenarios
# ===========================================================================

WHITE_RIBBON_EXCEPTIONS = [
    {
        "step_index": 1,
        "exception_type": "NetworkTimeout",
        "description": "Entity Retriever API returns 404 timeout when querying director information",
    },
]

WHITE_RIBBON_RECOVERY = {
    "NetworkTimeout_step1": "Retry: re-call EntityRetriever with same parameters 'Who directed The White Ribbon?'",
}

# ===========================================================================
# "Miss Chris Audio Request" — financial MAS configuration
# ===========================================================================

FINANCE_MAS = MASConfig(
    description="Multi-agent system for financial professionals handling daily issues",
    executors=[
        ExecutorDef(
            name="AudioMessageAgent",
            capabilities=[AgentCapability.AUDIO_PROCESSING],
            description="Transcribes and parses audio messages to extract user intent",
            tools=["audio_transcription", "fc2_authentication", "file_access"],
        ),
        ExecutorDef(
            name="FileManagerAgent",
            capabilities=[AgentCapability.FILE_MANAGEMENT],
            description="Lists, accesses, and manages file directories",
            tools=["file_listing", "file_read", "directory_navigation"],
        ),
        ExecutorDef(
            name="StructuredDataManager",
            capabilities=[AgentCapability.STRUCTURED_DATA],
            description="Parses structured data files (CSV, Excel) and extracts relevant records",
            tools=["csv_parser", "data_filtering", "statistical_summary"],
        ),
        ExecutorDef(
            name="TextReadingAgent",
            capabilities=[AgentCapability.TEXT_READING],
            description="Reads and summarizes text documents (PDF, DOCX, TXT)",
            tools=["pdf_reader", "text_extraction", "key_point_summarization"],
        ),
        ExecutorDef(
            name="QuantitativeFinancier",
            capabilities=[AgentCapability.QUANTITATIVE],
            description="Performs quantitative financial calculations and analysis",
            tools=["ratio_calculation", "percentage_computation", "financial_modeling"],
        ),
        ExecutorDef(
            name="SummaryAgent",
            capabilities=[AgentCapability.SUMMARIZATION],
            description="Generates comprehensive work reports and final conclusions",
            tools=["report_generation", "conclusion_synthesis", "formatting"],
        ),
        ExecutorDef(
            name="GUIOperator",
            capabilities=[AgentCapability.GUI_OPERATION],
            description="Handles GUI operations like dashboard updates and report publishing",
            tools=["app_operation", "dashboard_update", "report_publishing"],
        ),
    ],
    max_steps=20,
)

# ===========================================================================
# "Miss Chris Audio Request" — seed data
# ===========================================================================

FINANCE_SEED = SeedData(
    query=(
        "Audio file 'questuinc_4ca4238.wav': I wonder what percentage of wholesale "
        "distribution channels are due to Europe as of March 31, 2018? "
        "Files are in directory c4ca4238."
    ),
    answer="40.31%",
    domain="financial_analysis",
    metadata={
        "europe_value": 4928,
        "total_value": 12226,
        "percentage": "40.31%",
        "directory": "c4ca4238",
        "files": ["data.csv", "report.pdf"],
    },
)


# ===========================================================================
# Demo / test functions
# ===========================================================================


def demo_white_ribbon():
    """Demonstrate the full IWG pipeline on the White Ribbon case study.

    This runs the complete pipeline:
      Seed → Scout → Wrapper → Assemble → [Validation] → Evaluate
    """
    print("=" * 70)
    print("IWG Pipeline Demo — 'The White Ribbon' Case Study")
    print("=" * 70)

    # ---- Step 1: Create pipeline ----
    pipeline = IWGPipeline(mas_config=WHITE_RIBBON_MAS)

    # ---- Step 2: Scout (rule-based, no LLM needed) ----
    print("\n[1] Scout Agent: Inverse Planning")
    print("-" * 40)
    scout = ScoutAgent()
    plan = scout.plan_rule_based(WHITE_RIBBON_SEED, WHITE_RIBBON_MAS)
    print(f"  Reasoning approach: rule-based inverse analysis")
    print(f"  Task Marks generated: {len(plan.task_marks)}")
    for m in plan.task_marks:
        ext = " [EXTENDED]" if m.is_extension else ""
        print(f"    {m.id}: {m.description} → {m.assigned_agent}{ext}")
        print(f"         deps={m.dependencies}, cp_hint='{m.checkpoint_hint}'")

    # ---- Step 3: Wrapper (rule-based) ----
    print("\n[2] Wrapper Agent: Environment Synthesis")
    print("-" * 40)
    wrapper = WrapperAgent()
    wrapper_out = wrapper.synthesize_rule_based(plan)
    print(f"  Environments synthesized: {len(wrapper_out.environments)}")
    for ei in wrapper_out.environments:
        cp_val = ei.checkpoint.expected_value if ei.checkpoint else "(none)"
        print(f"    {ei.id} [{ei.agent_name}]: cp='{cp_val}'")

    # ---- Step 4: Assemble instance ----
    instance = BenchmarkInstance(
        seed_data=WHITE_RIBBON_SEED,
        gold_agent_sequence=[m.assigned_agent for m in plan.task_marks],
        gold_checkpoints=[ei.checkpoint for ei in wrapper_out.environments if ei.checkpoint],
        environments=wrapper_out.environments,
        exception_scenarios=wrapper_out.exception_scenarios,
        gold_recovery_plans=wrapper_out.gold_recovery_plans,
    )
    print(f"\n[3] Assembled Benchmark Instance: {instance.id}")
    print(f"  Gold agent sequence: {' → '.join(instance.gold_agent_sequence)}")
    print(f"  Checkpoints: {len(instance.gold_checkpoints)}")
    print(f"  Exception scenarios: {len(instance.exception_scenarios)}")

    # ---- Step 5: Print the gold standard ----
    print("\n[4] Gold Standard Workflow (from paper Figure 2)")
    print("-" * 40)
    print(f"  Query: {WHITE_RIBBON_SEED.query}")
    print(f"  Answer: {WHITE_RIBBON_SEED.answer}")
    print()
    for ei in WHITE_RIBBON_ENVIRONMENTS:
        print(f"  Step {ei.step_index} [{ei.agent_name}]:")
        print(f"    Prompt: {ei.tool_prompt}")
        print(f"    Output: {ei.tool_output[:100]}...")
        if ei.checkpoint:
            print(f"    [checkpoint_{ei.step_index}] expected: '{ei.checkpoint.expected_value}'")
        print()

    # ---- Step 6: Show metrics computation on a simulated trajectory ----
    print("\n[5] Metrics Demonstration (simulated trajectory)")
    print("-" * 40)
    print(f"  Gold agent sequence: {instance.gold_agent_sequence}")
    print(f"  Simulated pred sequence: {instance.gold_agent_sequence}")
    from .metrics import lcs_f1, task_success
    lcs = lcs_f1(instance.gold_agent_sequence, instance.gold_agent_sequence)
    print(f"  LCS-F1 (perfect match): {lcs['f1']:.4f}")

    # Simulate partial match (missing step M_2)
    partial = instance.gold_agent_sequence[:2] + instance.gold_agent_sequence[3:]
    lcs_partial = lcs_f1(instance.gold_agent_sequence, partial)
    print(f"  LCS-F1 (skipped M_2): {lcs_partial['f1']:.4f}")

    # Task Success
    cp_values = [cp.expected_value for cp in instance.gold_checkpoints]
    ts = task_success(instance.gold_checkpoints, cp_values)
    print(f"  Task Success (all matched): {ts}")

    # With one wrong checkpoint
    wrong_cps = cp_values.copy()
    wrong_cps[1] = "Steven Spielberg"
    ts_wrong = task_success(instance.gold_checkpoints, wrong_cps)
    print(f"  Task Success (one wrong): {ts_wrong}")

    print("\n" + "=" * 70)
    print("Demo complete. The IWG pipeline is ready for LLM integration.")
    print("=" * 70)

    return instance


def demo_finance_case():
    """Demonstrate the financial MAS orchestration case from the appendix."""
    print("=" * 70)
    print("IWG Pipeline Demo — 'Miss Chris Audio Request' Case Study")
    print("=" * 70)

    pipeline = IWGPipeline(mas_config=FINANCE_MAS)
    scout = ScoutAgent()
    plan = scout.plan_rule_based(FINANCE_SEED, FINANCE_MAS)

    print(f"\nTask Marks for financial workflow:")
    for m in plan.task_marks:
        ext = " [EXTENDED]" if m.is_extension else ""
        print(f"  {m.id}: {m.description} → {m.assigned_agent}{ext}")

    print(f"\nThis mirrors the paper's Appendix workflow:")
    print(f"  Step 1: AudioMessageAgent → transcribe audio")
    print(f"  Step 2: FileManagerAgent → list files")
    print(f"  Step 3: StructuredDataManager + TextReadingAgent → parse data (parallel)")
    print(f"  Step 4: QuantitativeFinancier → compute 40.31%")
    print(f"  Step 5: SummaryAgent → generate report")

    return plan


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("IWG — Inverse Workflow Generation Pipeline")
    print("Based on: Recognize Your Orchestrator (ICML 2026)\n")

    demo_white_ribbon()
    print("\n\n")
    demo_finance_case()
