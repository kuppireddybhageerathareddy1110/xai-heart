def counterfactual_to_text(original, cf):

    if cf is None:
        return "No safe clinical modification found within physiological limits."

    changes = []

    for col in original.columns:
        old = float(original[col])
        new = float(cf[col])

        if abs(old-new) > 0.05*max(1,abs(old)):

            if new < old:
                changes.append(f"reduce {col} to {new:.1f}")
            else:
                changes.append(f"increase {col} to {new:.1f}")

    if not changes:
        return "Risk already minimal."

    return "Risk reduces to low if patient can: " + ", ".join(changes) + "."
