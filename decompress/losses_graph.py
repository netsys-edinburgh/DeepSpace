'''
/*
    * This file is part of DeepSpace.
    *
    * DeepSpace is free software: you can redistribute it and/or modify
    * it under the terms of the GNU Affero General Public License as published by
    * the Free Software Foundation, either version 3 of the License, or
    * (at your option) any later version.
    *
    * DeepSpace is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    * GNU Affero General Public License for more details.
    *
    * You should have received a copy of the GNU Affero General Public License
    * along with DeepSpace.  If not, see <https://www.gnu.org/licenses/>.
    */
'''
import matplotlib.pyplot as plt
import numpy as np
import re

with open('./losses.txt', 'r') as file:
    loss_data = file.read()

#delete blank lines
loss_data = '\n'.join(line for line in loss_data.split('\n') if line.strip())


# Parse loss data and extract generator and discriminator loss
lines = loss_data.strip().split('\n')
generator_loss = [float(line.split(', ')[1].split(': ')[1]) for line in lines]
discriminator_loss = [float(line.split(', ')[2].split(': ')[1]) for line in lines]

iterations = [int(match.group(1)) for match in re.finditer(r'iteration(\d+)', loss_data)]
num_iterations = len(iterations)  # Adjust this based on your needs

iterations = [100 + 100 * i for i in range(num_iterations)]


# Plotting the generator and discriminator loss
plt.figure(figsize=(10, 6))
plt.plot(iterations, generator_loss, label='Generator Loss', marker='o', linestyle='-', markersize=2.5)
plt.plot(iterations, discriminator_loss, label='Discriminator Loss', marker='o', linestyle='-', markersize=2.5)
plt.title('Generator and Discriminator Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
