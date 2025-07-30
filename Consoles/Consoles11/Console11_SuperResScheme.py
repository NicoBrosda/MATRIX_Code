import matplotlib.pyplot as plt

# Example arrays
array1 = ['p1', 'p2', 'p3', 'p4']  # N=4
array2 = ['g1', 'g2', 'g3', 'g4']
N = len(array1)
box_size = 1

fig, ax = plt.subplots(figsize=(2*N, 3))

# Top row: N big boxes starting at x=0
for i in range(N):
    rect = plt.Rectangle((i*box_size, 2), box_size, box_size,
                         fill=False, edgecolor='black')
    ax.add_patch(rect)
    ax.text(i*box_size + box_size/2, 2 + box_size/2, array1[i],
            va='center', ha='center', fontsize=12, color='blue')

# Middle row: 2N+2 small boxes starting at x=0, width=box_size/2
num_small_boxes = 2*N + 2
for i in range(num_small_boxes):
    rect = plt.Rectangle((i*(box_size/2), 1), box_size/2, box_size/2,
                         fill=False, edgecolor='black')
    ax.add_patch(rect)
    ax.text(i*(box_size/2) + (box_size/4), 1 + (box_size/4),
            f'h{i}', va='center', ha='center', fontsize=10, color='purple')

# Bottom row: N big boxes shifted by half box_size → start at x=box_size/2
for i in range(N):
    x = i*box_size + box_size/2
    rect = plt.Rectangle((x, 0), box_size, box_size,
                         fill=False, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + box_size/2, 0 + box_size/2, array2[i],
            va='center', ha='center', fontsize=12, color='green')

# Add row labels on the left
ax.text(-0.7, 2.5, 'Array 1', fontsize=12, va='center')
ax.text(-0.7, 1.25, 'Super-res', fontsize=12, va='center')
ax.text(-0.7, 0.5, 'Array 2', fontsize=12, va='center')

# Adjust axes
total_width = N*box_size + box_size/2  # since bottom row starts half box later
ax.set_xlim(-1, total_width+0.5)
ax.set_ylim(-0.5, 3.2)
ax.axis('off')

plt.title('Top row aligned, bottom row shifted by half box; middle row has h0 and hN+1 at edges')
plt.show()
