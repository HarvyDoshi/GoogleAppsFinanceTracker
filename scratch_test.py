import sys
import os
sys.path.append(os.path.abspath('trace'))
from environments.trace_env.tools.gmail_tool import search_gmail_with_attachments
from environments.trace_env.tools.transaction_parser import parse_transaction, extract_total

emails = search_gmail_with_attachments('newer_than:10d', max_results=10, analyse_images=False)
for email in emails:
    subj = email.get('subject', '')
    if 'Tax Invoice' in subj or 'Uber' in subj or 'Rapido' in subj:
        tx = parse_transaction(email)
        print(f'\n--- {subj} ---')
        print(f'Total extracted: {tx.get("total")}')
        body = email.get('body_text', '') or email.get('snippet', '')
        print('Body len:', len(body))
        
        # Test extract_total directly
        total = extract_total(subj + ' ' + body)
        print(f'extract_total direct call: {total}')
        
        if not total and len(body) > 100:
            idx = body.find('Total')
            if idx != -1:
                print('Text around Total:', repr(body[max(0, idx-50):idx+150]))
            else:
                print('No Total found in body')
