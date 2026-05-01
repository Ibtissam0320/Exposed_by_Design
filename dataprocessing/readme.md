# Knowledge graphs

## Download

### NELL
- Source: https://rtw.ml.cmu.edu/rtw/
- Use the processed version with tab-separated triples (`h\tr\tt`)

### FB15k-237
- Source: https://www.microsoft.com/en-us/download/details.aspx?id=52312
- Or via HuggingFace: https://huggingface.co/datasets/KGraph/FB15k-237

### HealthKG
- Source: https://github.com/Boreico/KGE_QCB_Project/tree/main/Phase%205%20-%20Entity%20Definition

---

## Splitting Public / Sensitive Triples

A script `split.py` is provided to split the raw KG into a public graph and sensitive files (one per relation):

\```bash
python scripts/split.py \
  --global_path /path/to/full_kg.tsv \
  --relation "sensitive_relation_1" \
  --relation "sensitive_relation_2" \
  --outdir /path/to/output/
\```

---

## Target Relations

### Attack1 (Link Inference)

| Dataset   | Sensitive Relation                                                            |
|-----------|--------------------------------------------------------------------------------|
| NELL      | `concept:teamplaysagainstteam`                                                 |
| FB15k-237 | `sports__sports_position__players.__sports__sports_team_roster__team`         |
| HealthKG  | `has_taxonomy`                                                                 |

### Attack3 (Graph Reconstruction)

**FB15k-237:**
\```
education__educational_institution__students_graduates.__education__education__student,
film__film__genre,
people__person__profession,
sports__sports_position__players.__sports__sports_team_roster__team
\```

**HealthKG:**
\```
has_age_category,
has_age_living_apart,
has_family_ID,
has_gender,
has_is_westernized,
has_is-from,
has_zygosity
\```

**NELL:**
\```
concept:atlocation,
concept:proxyfor,
concept:subpartof,
concept:teamplaysagainstteam
\```

## Utility (Link Predicition)
For utility evaluation, we performed link prediction tasks on both the original public knowledge graphs and the defended graphs, focusing on specific relation types. An 80/20 split was used for training and testing.

- **NELL**: `concept:athleteplayssport`  
- **HealthKG**: `has_age_category`  
- **FB15K**: `/people/person/nationality`  

All results are reproducible by running the `utility` script located in the `experiments` folder.