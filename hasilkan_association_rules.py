import association_rules

def hasilkan_association_rules(frequent_itemsets, min_confidence, min_lift):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]

    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: frozenset(x))

        for _, rule in rules.iterrows():
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            antecedent_support = frequent_itemsets[frequent_itemsets['itemsets'] == antecedents]['support'].values[0]
            consequent_support = frequent_itemsets[frequent_itemsets['itemsets'] == consequents]['support'].values[0]

            if antecedents.isdisjoint(consequents):
                combined_support = antecedent_support + consequent_support
                support_AB = frequent_itemsets[(frequent_itemsets['itemsets'] == antecedents.union(consequents))]['support'].values[0]

                if support_AB > 0:
                    confidence = combined_support / support_AB
                    lift = confidence / antecedent_support

                    rules.at[_, 'antecedent support'] = antecedent_support
                    rules.at[_, 'consequent support'] = consequent_support
                    rules.at[_, 'support'] = combined_support
                    rules.at[_, 'confidence'] = confidence
                    rules.at[_, 'lift'] = lift

                    leverage = combined_support - (antecedent_support * consequent_support)
                    conviction = (1 - consequent_support) / (1 - confidence)

                    rules.at[_, 'leverage'] = leverage
                    rules.at[_, 'conviction'] = conviction

        rules['zhangs_metric'] = (rules['confidence'] - rules['lift']) / (rules['confidence'] + rules['lift'])

    return rules