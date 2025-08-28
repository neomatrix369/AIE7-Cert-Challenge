#!/usr/bin/env python3
"""
Batch process CSV files with questions through backend API
Usage: python batch_process_questions.py [input_csv] [output_prefix]
"""
import pandas as pd
import requests
import time
import sys
import os
from pathlib import Path


def process_questions_csv(input_file: str, output_prefix: str = "gap-analysis-answers", 
                         backend_url: str = "http://localhost:8000"):
    """
    Process CSV file with questions through backend /ask API
    Supports resuming from partially completed runs
    
    Args:
        input_file: Path to input CSV with questions
        output_prefix: Prefix for output filename 
        backend_url: Backend API base URL
    """
    # Read input CSV
    try:
        df = pd.read_csv(input_file)
        print(f"📄 Loaded {len(df)} questions from {input_file}")
    except Exception as e:
        print(f"❌ Error reading {input_file}: {e}")
        return
    
    # Identify required columns
    question_col = None
    role_col = None
    focus_col = None
    
    for col in df.columns:
        if 'question' in col.lower() and 'text' in col.lower():
            question_col = col
        elif col.lower() == 'role':
            role_col = col
        elif 'focus' in col.lower():
            focus_col = col
    
    if not question_col:
        print("❌ No question column found (looking for column containing 'question' and 'text')")
        return
    
    print(f"🔍 Using columns:")
    print(f"   - Question: '{question_col}'")
    print(f"   - Role: '{role_col}'" if role_col else "   - Role: Not found")
    print(f"   - Focus: '{focus_col}'" if focus_col else "   - Focus: Not found")
    
    # Generate output filename in same directory as input file
    input_path = Path(input_file)
    input_name = input_path.stem
    input_dir = input_path.parent
    output_file = input_dir / f"{output_prefix}-{input_name}.csv"
    
    existing_results = []
    start_idx = 0
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            start_idx = len(existing_results)
            print(f"📂 Found existing results: {start_idx} questions already processed")
            print(f"🔄 Resuming from question {start_idx + 1}")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")
            print("🔄 Starting fresh...")
    
    # Prepare results from existing data
    results = existing_results.copy()
    
    # Process remaining questions
    remaining_df = df.iloc[start_idx:]
    
    if len(remaining_df) == 0:
        print("✅ All questions already processed!")
        return
    
    print(f"🚀 Processing {len(remaining_df)} remaining questions...")
    print("=" * 60)
    
    # Process each remaining question with progress tracking
    for idx, row in remaining_df.iterrows():
        question = row[question_col]
        role = row.get(role_col, '') if role_col else ''
        focus = row.get(focus_col, '') if focus_col else ''
        
        # Format enhanced query with role and focus
        enhanced_query = question
        if role or focus:
            prefix_parts = []
            if role:
                prefix_parts.append(f"Role: {role}")
            if focus:
                prefix_parts.append(f"Focus: {focus}")
            enhanced_query = f"{' '.join(prefix_parts)} {question}"
        
        current_pos = idx - start_idx + 1
        total_remaining = len(remaining_df)
        
        # Progress indicator
        progress_bar = "█" * int(20 * current_pos / total_remaining) + "░" * (20 - int(20 * current_pos / total_remaining))
        print(f"\n[{progress_bar}] {current_pos}/{total_remaining} ({(current_pos/total_remaining)*100:.1f}%)")
        print(f"🤔 Q{idx+1}: {question[:60]}...")
        if role or focus:
            print(f"   👤 Role: {role} | 🎯 Focus: {focus}")
        
        try:
            # Call backend API with enhanced query
            response = requests.post(
                f"{backend_url}/ask",
                json={"question": enhanced_query},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', 'No answer received')
                sources_count = data.get('sources_count', 0)
                success = data.get('success', False)
                
                results.append({
                    'response': answer,
                    'question': question,
                    'sources_count': sources_count,
                    'success': success,
                    'original_row_id': idx
                })
                print(f"   ✅ Success - {sources_count} sources, {len(answer)} chars")
                
            else:
                print(f"   ❌ API Error {response.status_code}: {response.text}")
                results.append({
                    'response': f"API Error {response.status_code}: {response.text}",
                    'question': question,
                    'sources_count': 0,
                    'success': False,
                    'original_row_id': idx
                })
        
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            results.append({
                'response': f"Request failed: {e}",
                'question': question,                
                'sources_count': 0,
                'success': False,
                'original_row_id': idx
            })
        
        # Save progress every 5 questions (in case of interruption)
        if current_pos % 5 == 0:
            temp_df = pd.DataFrame(results)
            try:
                temp_df.to_csv(output_file, index=False)
                print(f"   💾 Progress saved ({current_pos}/{total_remaining} completed)")
            except Exception:
                pass  # Continue if save fails
        
        # Rate limiting - avoid overwhelming the API
        time.sleep(1)
    
    # Final save
    results_df = pd.DataFrame(results)
    
    try:
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Final results saved to: {output_file}")
        print(f"📊 Summary: {results_df['success'].sum()}/{len(results_df)} successful responses")
        print(f"📈 Total processed: {len(results_df)}/{len(df)} questions completed")
        
        if len(results_df) < len(df):
            print(f"🔄 {len(df) - len(results_df)} questions remaining - run again to continue")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")


def main():
    """Main entry point with command line argument handling"""
    if len(sys.argv) < 2:
        print("Usage: python batch_process_questions.py <input_csv> [output_prefix]")
        print("Example: python batch_process_questions.py .questions/weak_questions_2025-08-25.csv")
        return
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "gap-analysis-answers-to"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print("🚀 Starting batch processing...")
    print(f"📄 Input: {input_file}")
    print(f"🏷️  Output prefix: {output_prefix}")
    print("🌐 Backend: http://localhost:8000")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend health check failed")
    except Exception:
        print("❌ Backend not reachable - make sure it's running on localhost:8000")
        return
    
    process_questions_csv(input_file, output_prefix)


if __name__ == "__main__":
    main()