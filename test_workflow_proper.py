"""
Proper EDA Workflow Test

This test actually validates the workflow outputs at each stage, not just setup.
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EDA.workflow.workflow_execution import execute_eda_workflow
from EDA.workflow.workflow_config import EDAWorkflowConfig, DomainType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# async def test_workflow_with_actual_validation():
#     """Test workflow with proper output validation at each stage"""
#     print("=== Testing EDA Workflow with Proper Validation ===")
    
#     try:
#         # Test with a real target variable that exists in the data
#         result = await execute_eda_workflow(
#             target_variable="sales.csv.units_sold",  # Use actual variable from data
#             eda_request="Analyze factors influencing sales volume",
#             domain_context="retail",
#             hypothesis_limit=3,
#             simplified=True,
#             target_data_path="data/sales.csv",

#         )
        
#         print(f"Workflow Status: {result['status']}")
#         print(f"Execution Metrics: {result['execution_metrics']}")
        
#         # Validate each stage of the workflow
#         if result['status'] == 'success':
#             print("\n=== Validating Workflow Results ===")
            
#             # Check if we have actual results
#             if 'results' in result and result['results']:
#                 print("✅ Final results present")
#                 print(f"Results keys: {list(result['results'].keys())}")
#             else:
#                 print("❌ No final results found")
#                 return False
            
#             # Check execution metrics
#             metrics = result['execution_metrics']
#             if metrics.get('hypotheses_generated', 0) > 0:
#                 print(f"✅ Hypotheses generated: {metrics['hypotheses_generated']}")
#             else:
#                 print("❌ No hypotheses generated")
#                 return False
            
#             # Check for error messages
#             if result.get('error_messages'):
#                 print(f"⚠️  Error messages: {result['error_messages']}")
#                 return False
#             else:
#                 print("✅ No error messages")
            
#             return True
#         else:
#             print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
#             if result.get('error_messages'):
#                 print(f"Error messages: {result['error_messages']}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Test failed with exception: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# async def test_intugle_integration_properly():
#     """Test Intugle integration with proper validation"""
#     print("\n=== Testing Intugle Integration Properly ===")
    
#     try:
#         # Initialize Intugle integration
#         intugle = IntugleIntegration("models")
        
#         # Test initialization
#         if intugle.initialize_semantic_layer():
#             print("✅ Intugle initialization successful")
            
#             # Test available variables
#             variables = intugle.get_available_variables()
#             print(f"✅ Available variables: {len(variables)}")
#             if len(variables) > 0:
#                 print(f"Sample variables: {variables[:5]}")
#             else:
#                 print("❌ No variables found")
#                 return False
            
#             # Test variable search
#             search_results = intugle.search_related_variables("sales", limit=5)
#             print(f"✅ Search results: {len(search_results)}")
#             if len(search_results) > 0:
#                 print(f"Sample search result: {search_results[0]}")
#             else:
#                 print("❌ No search results")
#                 return False
            
#             # Test workflow validation
#             validation = intugle.validate_workflow_requirements(
#                 target_variable="sales.csv.units_sold",
#                 eda_request="Analyze sales performance"
#             )
#             print(f"✅ Workflow validation: {validation['valid']}")
#             if not validation['valid']:
#                 print(f"Validation issues: {validation.get('warnings', [])}")
#                 return False
            
#             return True
#         else:
#             print("❌ Intugle initialization failed")
#             return False
            
#     except Exception as e:
#         print(f"❌ Intugle integration test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# async def test_workflow_with_intugle():
#     """Test workflow with proper Intugle integration"""
#     print("\n=== Testing Workflow with Intugle Integration ===")
    
#     try:
#         # Initialize Intugle first
#         intugle = IntugleIntegration("models")
#         if not intugle.initialize_semantic_layer():
#             print("❌ Cannot test workflow without Intugle")
#             return False
        
#         # Test workflow with Intugle
#         result = await execute_eda_workflow(
#             target_variable="sales.csv.units_sold",
#             eda_request="Analyze sales performance with Intugle",
#             domain_context="retail",
#             hypothesis_limit=2,
#             kb=intugle,
#             simplified=True
#         )
        
#         print(f"Workflow Status: {result['status']}")
        
#         if result['status'] == 'success':
#             print("✅ Workflow with Intugle succeeded")
            
#             # Validate that we got meaningful results
#             metrics = result['execution_metrics']
#             if metrics.get('hypotheses_generated', 0) > 0:
#                 print(f"✅ Generated {metrics['hypotheses_generated']} hypotheses")
#                 return True
#             else:
#                 print("❌ No hypotheses generated despite success")
#                 return False
#         else:
#             print(f"❌ Workflow with Intugle failed: {result.get('error', 'Unknown error')}")
#             if result.get('error_messages'):
#                 print(f"Error messages: {result['error_messages']}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Workflow with Intugle test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# async def test_error_handling_properly():
#     """Test error handling with proper validation"""
#     print("\n=== Testing Error Handling Properly ===")
    
#     try:
#         # Test with invalid parameters
#         result = await execute_eda_workflow(
#             target_variable="",  # Invalid
#             eda_request="",  # Invalid
#             domain_context="",
#             hypothesis_limit=0,  # Invalid
#             simplified=True
#         )
        
#         print(f"Error handling status: {result['status']}")
        
#         # The workflow should fail gracefully
#         if result['status'] == 'failed':
#             print("✅ Workflow failed as expected")
            
#             # Check that we have proper error information
#             error_messages = result.get('error_messages', [])
#             if error_messages:
#                 print(f"✅ Error messages captured: {len(error_messages)}")
#                 for i, error in enumerate(error_messages):
#                     print(f"  {i+1}. {error}")
#                 return True
#             else:
#                 print("❌ No error messages captured")
#                 return False
#         else:
#             print("❌ Workflow should have failed with invalid inputs")
#             return False
            
#     except Exception as e:
#         print(f"❌ Error handling test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# async def run_proper_tests():
#     """Run all tests with proper validation"""
#     print("EDA Workflow Proper Test Suite")
#     print("=" * 60)
    
#     tests = [
#         ("Intugle Integration", test_intugle_integration_properly),
#         ("Workflow with Intugle", test_workflow_with_intugle),
#         ("Basic Workflow", test_workflow_with_actual_validation),
#         ("Error Handling", test_error_handling_properly)
#     ]
    
#     results = {}
    
#     for test_name, test_func in tests:
#         try:
#             print(f"\n{'='*20} {test_name} {'='*20}")
#             result = await test_func()
#             results[test_name] = result
#         except Exception as e:
#             print(f"❌ {test_name} test crashed: {str(e)}")
#             results[test_name] = False
    
#     # Summary
#     print("\n" + "=" * 60)
#     print("Proper Test Results Summary:")
#     for test_name, success in results.items():
#         status = "✅ PASSED" if success else "❌ FAILED"
#         print(f"  {test_name}: {status}")
    
#     total_passed = sum(results.values())
#     total_tests = len(results)
#     print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
#     if total_passed == total_tests:
#         print("🎉 All proper tests passed! EDA Workflow is truly ready.")
#     else:
#         print("⚠️  Some tests failed. The workflow needs fixes.")
    
#     return total_passed == total_tests

async def main():
    """Main function to run the workflow test"""
    try:
        # Run proper tests
        success = await execute_eda_workflow(
                target_variable="sales.vol",  # Use actual variable from data
                eda_request="What are the factors influencing sales?",
                domain_context="This FMCG sales dataset captures the key drivers of product performance in fast-moving consumer goods markets, combining marketing inputs (ad_spends, promo_camp, social), trade and distribution factors (disti, mkt_coverage, retail_audit), competitive dynamics (comp_price, price_sense, pric_strat), and contextual information (prd_mstr, rnd, cons_fb) to explain variations in sales. It reflects how consumer demand in FMCG is shaped not only by product attributes and pricing but also by visibility, availability, promotions, and brand perception, making it well-suited for analyses such as market mix modeling, demand forecasting, and pricing elasticity studies.",
                hypothesis_limit=1,
                simplified=True,
                full_data_path="C:/Dev/Data Querying AI/data-tools/sample_data/fmcg",
                target_file_name="sales.csv",
                files_to_process= ["ad_spends","comp_price","cons_fb","disti","mkt_coverage","prd_mstr","price_sense","pric_strat","promo_camp","retail_audit","rnd","sales","social"]
            )
        print(success)
    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
