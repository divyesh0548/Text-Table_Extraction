from fixed_pattern_alignment import pattern_based_alignment_fixed

# Run your corrected approach
result = pattern_based_alignment_fixed('bs3.pdf', 'payments_final.xlsx', debug=True)
print(f'Successfully extracted {len(result)} payment records!')
