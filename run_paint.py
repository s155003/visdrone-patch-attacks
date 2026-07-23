"""
Full-body PAINT attack: the entire car body silhouette is painted with ONE continuous
adversarial pattern, excluding ALL glass (front + rear windshields + side windows).
Like a complete adversarial vinyl wrap / paint job. car->van, 100 images, GPU.
  R51 (paint): full body-minus-glass adversarial coverage
"""
import patch_attack_paint as PT
PT.USE_PAINT = True
PT.SAVE_DIR = "patch_examples_run51_paint"
print("\n"+"#"*70+"\n#  R51 PAINT: full body silhouette minus all glass\n"+"#"*70)
PT.run()
