from encodeval.system_ranking import get_results

average_scores, system_ranking = get_results(
    base_path="./results",
    models=["EuroBERT-210m"],
    task_type="SC",
    dataset="xnli",
    valid_langs=["en"],
)

print("Average scores type:", type(average_scores))
print("Average scores:")
print(average_scores)
print("\nSystem ranking type:", type(system_ranking))
print("System ranking:")
print(system_ranking)
