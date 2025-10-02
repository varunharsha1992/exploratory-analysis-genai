"""
Test script specifically for DataProductBuilder functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from EDA.tools.intugle_agent_tools import *

def test_dataproduct_builder():
    """Test DataProductBuilder functionality"""
    
    print('=== TESTING DATAPRODUCT BUILDER ===')
    print(f'Intugle available: {is_intugle_available()}')
    
    if not is_intugle_available():
        print('❌ Intugle tools not available')
        return
    
    # Test 1: Get DataProduct schema
    print('\n1. Getting DataProduct schema...')
    etl_schema = get_dataproduct_etl_schema('customer analysis')
    if etl_schema.get('status') == 'success':
        dp_config = etl_schema.get('data_product_config', {})
        print(f'  ✓ Schema generated successfully')
        print(f'    Name: {dp_config.get("name", "N/A")}')
        print(f'    Fields: {len(dp_config.get("fields", []))}')
        
        # Test 2: Build DataProduct
        print('\n2. Building DataProduct...')
        build_result = build_dataproduct(dp_config)
        if build_result.get('status') in ['success', 'partial_success']:
            print(f'  ✓ DataProduct build completed')
            build_info = build_result.get('build_result', {})
            print(f'    Status: {build_info.get("status", "unknown")}')
            print(f'    Message: {build_info.get("message", "N/A")}')
            if 'fields_processed' in build_info:
                print(f'    Fields processed: {build_info.get("fields_processed", 0)}')
        else:
            print(f'  ✗ Build failed: {build_result.get("error", "Unknown error")}')
    else:
        print(f'  ✗ Schema generation failed: {etl_schema.get("error", "Unknown error")}')
    
    print('\n✅ DataProductBuilder test completed!')

if __name__ == "__main__":
    test_dataproduct_builder()
