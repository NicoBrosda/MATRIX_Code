import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *
import cv2


# Get the correct mapping for the matrix array
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

for mapping in ['../../Files/Mapping_MatrixArray.xlsx', '../../Files/Mapping_BigMatrix_2.xlsx',
                '../../Files/Mapping_SmallMatrix1.xlsx', '../../Files/Mapping_SmallMatrix2.xlsx']:
    mapping = Path(mapping)
    print(mapping.name)

    data2 = pd.read_excel(mapping, header=None)
    mapping_map = data2.to_numpy().flatten()
    translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

    # Example: create a dummy 11x11 assignment (replace with your actual mapping)
    # For instance, channels numbered 1–121
    # assignment = np.arange(1, 11*11 + 1).reshape((11, 11))
    assignment = translated_mapping.reshape((11, 11)).T[::-1, :] + 1
    print(assignment)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(assignment, cmap='viridis')

    # Annotate each pixel with its channel number
    for i in range(assignment.shape[0]):
        for j in range(assignment.shape[1]):
            ax.text(j, i, str(assignment[i, j]),
                    ha='center', va='center', color='white', fontsize=8)

    # Axis and layout
    ax.set_title("Diode → Measurement Channel Assignment", fontsize=14)
    ax.set_xlabel("X pixel index")
    ax.set_ylabel("Y pixel index")
    ax.set_xticks(np.arange(11))
    ax.set_yticks(np.arange(11))
    ax.set_xticklabels(np.arange(1, 12))
    ax.set_yticklabels(np.arange(1, 12))
    ax.invert_yaxis()  # Optional: if you want image-like coordinates

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Channel number')

    plt.tight_layout()
    format_save(Path('../../Files/'), f'{mapping.name[:-5]}', save_format='.pdf')
