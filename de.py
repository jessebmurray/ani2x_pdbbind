import matplotlib.pyplot as plt
from ani2x import load_df_gen

df_gen = load_df_gen()
df_gen[~df_gen.Binding_Symbol.isin({'>=', '<=', '='})].groupby('Binding_Symbol').pK.hist(density=False, histtype='step',
                                linewidth=2, legend=True, bins=15, alpha=0.7)

df_gen.groupby('Binding_Type').Binding_Symbol.value_counts()

df_gen.groupby('Binding_Type').pK.hist(histtype='step', density=True,
                                linewidth=2, legend=True, bins=20, alpha=0.7)

df_gen[df_gen.Refined][['Binding_Type', 'Binding_Symbol']].value_counts().to_frame()

df_gen.Binding_Type.value_counts()

df_gen.Binding_Symbol.value_counts()
