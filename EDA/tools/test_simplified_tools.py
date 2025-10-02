"""
Simple test script for ETL schema creation
"""
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from EDA.tools import intugle_agent_tools

config = {
    "target_variable": "sales.vol",
    "eda_request": "What are the factors influencing sales?",
    "domain_context": "This FMCG sales dataset captures the key drivers of product performance in fast-moving consumer goods markets, combining marketing inputs (ad_spends, promo_camp, social), trade and distribution factors (disti, mkt_coverage, retail_audit), competitive dynamics (comp_price, price_sense, pric_strat), and contextual information (prd_mstr, rnd, cons_fb) to explain variations in sales. It reflects how consumer demand in FMCG is shaped not only by product attributes and pricing but also by visibility, availability, promotions, and brand perception, making it well-suited for analyses such as market mix modeling, demand forecasting, and pricing elasticity studies.",
    "hypothesis_limit": 3,
    "simplified": False,
    "full_data_path": "C:/Dev/Data Querying AI/data-tools/sample_data/fmcg",
    "target_file_name": "sales.csv",
    "files_to_process": [
        "ad_spends", "comp_price", "cons_fb", "disti", "mkt_coverage",
        "prd_mstr", "price_sense", "pric_strat", "retail_audit",
        "rnd", "sales", "social"
    ]
}
intugle_tools = intugle_agent_tools.IntugleAgentTools(full_data_path=config['full_data_path'], files_to_process=config['files_to_process'])
def test_etl_schema():
    """Simple test function to check if ETL schema creation is working"""
    
    # Dummy request
    dummy_request = f"""
    {{
      "hypothesis_id": "hyp_1",
      "hypothesis": "Promotional campaigns have a positive impact on sales volume, reflected in an interaction effect between promotional spending and marketing coverage.",
      "target_variable": {{
        "name": "sales.vol",
        "alias": "sales_vol",
        "transformation": "none",
        "measure_func": "sum"
      }},
      "predictor_variable": {{
        "name": "promo_camp.sales_inc",
        "alias": "promo_sales_inc",
        "transformation": "interaction",
        "measure_func": "sum"
      }},
      "relationship_type": "interaction_effect",
      "expected_impact": "positive",
      "confidence": 0.87,
      "research_support": [
        "Promotions increase consumer engagement leading to higher sales.",
        "Marketing effectiveness has been shown to significantly influence FMCG sales."
      ],
      "interaction_features": [
        "promo_camp.sales_inc * mkt_coverage.st_cnt"
      ],
      "test_priority": 0.95,
      "aggregate_by": {{
        "date_sale": "sales.date",
        "product": "sales.prod_id"
      }},
      "data_requirements": {{
        "required_tables": [
          "sales",
          "promo_camp",
          "mkt_coverage"
        ],
        "required_columns": [
          "sales.vol",
          "promo_camp.sales_inc",
          "mkt_coverage.st_cnt"
        ],
        "join_requirements": "sales.prod_id = promo_camp.p_id AND sales.date = promo_camp.date"
      }}
    }}
    """
    
    print("Testing ETL schema creation...")
    print(f"Request: {dummy_request}")
    print("\n" + "="*80 + "\n")
    
    # Create ETL schema
    result = intugle_tools.create_etl_schema_with_llm(dummy_request)
    
    # Print result
    print("Result:")
    print(json.dumps(result, indent=2))
    
    # Check for errors
    if "error" in result:
        print("\n❌ ETL Schema creation failed!")
        print(f"Error: {result['error']}")
        return False
    
    if "etl_schema" in result:
        print("\n✅ ETL Schema created successfully!")
        print(f"Schema name: {result['etl_schema'].get('name', 'N/A')}")
        print(f"Number of fields: {len(result['etl_schema'].get('fields', []))}")
        return True
    
    return False

if __name__ == "__main__":
    test_etl_schema()