# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:08:21 2025

@author: Braulio
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# --------- Función de preprocesamiento ---------
def load_and_preprocess_image(path, target_size=(100, 100)):
    """
    Carga la imagen, la convierte a escala de grises, escala su rango a [0,255],
    y la redimensiona a `target_size`.
    """
    # Leer en escala de grises, como float64 en [0,1]
    img = imread(path, as_gray=True).astype(np.float64)

    # Redimensionar a target_size
    img = resize(img, target_size, anti_aliasing=True)

    # Llevar a [0,255]
    img = img * 255.0

    return img

# --------- Función auxiliar para ordenar grupos ---------
def sort_groups(groups, neighbors):
    """
    Ordena listas de grupos por su tamaño (descendente), devolviendo
    también la lista de vecinos correspondiente reordenada.
    """
    sizes = np.array([len(g) for g in groups])
    if len(sizes) > 1:
        new_groups = []
        new_neighbors = []
        for s in np.arange(np.max(sizes), np.min(sizes) - 1, -1):
            for i in np.where(sizes == s)[0]:
                new_groups.append(groups[i])
                new_neighbors.append(neighbors[i])
        return new_groups, new_neighbors
    return groups, neighbors


def persistence_entropy(lifetimes, label=""):
    """
    Calcula la entropía de persistencia de una lista de intervalos [b, d]
    y la entropía de Shannon normalizada. Imprime ambas.

    Parámetro:
        lifetimes: lista de longitudes de vida (death - birth) o lista de [birth, death].
        label: etiqueta para imprimir si se desea distinguir H0 o H1.

    Retorna:
        tuple: (entropía de persistencia, entropía de Shannon normalizada)
    """
    if not lifetimes:
        print(f"Entropía de {label}: 0.0 (sin intervalos)")
        return 0.0, 0.0

    # Detectar si vienen como [[b, d], ...] y convertir a [d - b]
    if isinstance(lifetimes[0], (list, tuple, np.ndarray)):
        lifetimes = [d - b for b, d in lifetimes]

    lifetimes = np.array(lifetimes, dtype=np.float64)
    lifetimes = lifetimes[lifetimes > 0]

    if lifetimes.size == 0:
        print(f"Entropía de {label}: 0.0 (sin intervalos positivos)")
        return 0.0, 0.0

    probs = lifetimes / np.sum(lifetimes)
    entropy = -np.sum(probs * np.log(probs))

    # Entropía de Shannon normalizada
    if len(probs) > 1:
        entropy_norm = entropy / np.log(len(probs))
    else:
        entropy_norm = 0.0

    return entropy, entropy_norm



# --------- Clase Topology (H₀ y H₁) ---------
class Topology:
    def __init__(self, image):
        """
        Inicializa la clase con la imagen dada. Convierte NaN a 255 (blanco)
        para que no formen parte de componentes H₀ oscuras, y calcula
        automáticamente los vecinos 8-conexos (para H₀) y 4-conexos (para H₁).
        Luego invoca el cálculo de grupos para H₀ y H₁.
        """
        # Convertimos NaN a 255 (para que no "nazcan" componentes H₀ en esos píxeles)
        self.img = np.nan_to_num(image, nan=255.0).astype(np.uint8)
        self.y, self.x = self.img.shape
        
        # Vector lineal de índices [0…x*y-1]
        self.n_pixel_flat = np.arange(self.x * self.y)
        # Matriz de índices con forma (y,x)
        self.n_pixel_img = self.n_pixel_flat.reshape(self.y, self.x).astype(int)
        
        # Píxeles frontera (para descartar agujeros que toquen bordes en H₁)
        self.limits = (
            list(self.n_pixel_img[:, 0]) +
            list(self.n_pixel_img[:, -1]) +
            list(self.n_pixel_img[0, :]) +
            list(self.n_pixel_img[-1, :])
        )
        
        # Máximo valor de intensidad en la imagen
        self.max_val = int(np.max(self.img))
        
        # Creamos una matriz con un contorno de -1 para calcular vecinos sin ruido
        mat = np.full((self.y + 2, self.x + 2), -1, dtype=int)
        mat[1:-1, 1:-1] = self.n_pixel_img
        
        # Generamos listas de vecinos para cada píxel:
        # - neighbors_0_pixels: 8-conectividad (para H₀)
        # - neighbors_1_pixels: 4-conectividad (para H₁)
        self.neighbors_0_pixels = []
        self.neighbors_1_pixels = []
        for j in range(1, self.y + 1):
            for i in range(1, self.x + 1):
                # Vecinos 8-conexos
                neigh0 = np.array([
                    mat[j-1, i-1], mat[j-1, i], mat[j-1, i+1],
                    mat[j,   i-1],            mat[j,   i+1],
                    mat[j+1, i-1], mat[j+1, i], mat[j+1, i+1]
                ], dtype=int)
                neigh0 = list(neigh0[neigh0 != -1])
                self.neighbors_0_pixels.append([int(n) for n in neigh0])
                
                # Vecinos 4-conexos
                neigh1 = np.array([
                    mat[j-1,   i],
                    mat[j,   i-1],        mat[j,   i+1],
                    mat[j+1,   i]
                ], dtype=int)
                neigh1 = list(neigh1[neigh1 != -1])
                self.neighbors_1_pixels.append([int(n) for n in neigh1])
        
        # Inicializamos estructuras que contendrán:
        # - groups_0, neighbors_0, lifetime_0, lifetime_0_extend (para H₀)
        # - groups_1, neighbors_1, lifetime_1, lifetime_1_extend (para H₁)
        self.groups_0 = []
        self.neighbors_0 = []
        self.lifetime_0 = []
        self.lifetime_0_extend = []
        self.groups_1 = []
        self.neighbors_1 = []
        self.lifetime_1 = []
        self.lifetime_1_extend = []
        
        # Calculamos automáticamente ambos (H₀ y H₁)
        self.calculate_groups_0()
        self.calculate_groups_1()

    def calculate_groups_0(self):
        """
        Calcula componentes conexas (H₀) por cada nivel de intensidad,
        registrando el intervalo [nacimiento, muerte] en lifetime_0_extend.
        """
        lifetime_0 = []
        
        for level in range(0, self.max_val + 1):
            # Seleccionamos todos los píxeles de intensidad EXACTA == level
            pixels = self.n_pixel_img[self.img == level]
            if len(pixels) == 0:
                continue
            # Encontramos grupos (componentes) para estos píxeles
            group_pixels, neighbor_pixels = self.connected_pixels(pixels, class_type=0)
            group_pixels, neighbor_pixels = sort_groups(group_pixels, neighbor_pixels)
            
            # Verificamos si cada grupo es nuevo o se fusiona con otro
            for i_g, group in enumerate(group_pixels):
                group_in_neighbors = []
                for i_v, neighbor in enumerate(self.neighbors_0):
                    # Si algún píxel de 'group' está en neighbors[i_v], se fusionan
                    for pixel in group:
                        if pixel in neighbor:
                            group_in_neighbors.append(i_v)
                            break
                if len(group_in_neighbors) == 0:
                    # Es un nuevo componente H₀
                    self.groups_0.append(group)
                    self.neighbors_0.append(neighbor_pixels[i_g])
                    lifetime_0.append(list(np.zeros(self.max_val + 1, dtype=int)))
                else:
                    # Se fusiona con componentes ya existentes (sobrevive el primero)
                    first = group_in_neighbors[0]
                    self.groups_0[first] += group
                    self.neighbors_0[first] += neighbor_pixels[i_g]
                    for n_g in group_in_neighbors[1:]:
                        self.groups_0[first] += self.groups_0[n_g]
                        self.neighbors_0[first] += self.neighbors_0[n_g]
                        self.groups_0[n_g] = []
                        self.neighbors_0[n_g] = []
            
            # Marcamos “1” en el vector de vida de cada componente que persista a este nivel
            for i_g, group in enumerate(self.groups_0):
                if len(group) > 0:
                    lifetime_0[i_g][level] = 1
        
        # Construimos lifetime_0_extend como [[birth, death], ...]
        self.lifetime_0_extend = [
            [np.min(np.where(life)[0]), np.max(np.where(life)[0]) + 1]
            for life in lifetime_0 if np.sum(life) > 0
        ]
        # lifetime_0 guarda la diferencia (persistencia) de cada intervalo
        self.lifetime_0 = list(np.diff(self.lifetime_0_extend, axis=1).flatten())

    def calculate_groups_1(self):
        """
        Calcula agujeros (H₁) recorriendo niveles de intensidad de mayor a menor,
        registrando intervalos [nacimiento, muerte] en lifetime_1_extend.
        """
        lifetime_1 = []
        
        for level in range(self.max_val, -1, -1):
            # Seleccionamos píxeles cuyo valor está entre (level, level+1]
            pixels = self.n_pixel_img[(self.img > level) & (self.img < level + 2)]
            if len(pixels) == 0:
                continue
            group_pixels, neighbor_pixels = self.connected_pixels(pixels, class_type=1)
            group_pixels, neighbor_pixels = sort_groups(group_pixels, neighbor_pixels)
            
            # Verificamos si cada agujero es nuevo o se fusiona
            for i_g, group in enumerate(group_pixels):
                group_in_neighbors = []
                for i_v, neighbor in enumerate(self.neighbors_1):
                    for pixel in group:
                        if pixel in neighbor:
                            group_in_neighbors.append(i_v)
                            break
                if len(group_in_neighbors) == 0:
                    # Un nuevo agujero H₁ (siempre que no toque el borde)
                    self.groups_1.append(group)
                    self.neighbors_1.append(neighbor_pixels[i_g])
                    lifetime_1.append(list(np.zeros(self.max_val + 1, dtype=int)))
                else:
                    # Se fusiona con un agujero ya existente
                    first = group_in_neighbors[0]
                    self.groups_1[first] += group
                    self.neighbors_1[first] += neighbor_pixels[i_g]
                    for n_g in group_in_neighbors[1:]:
                        self.groups_1[first] += self.groups_1[n_g]
                        self.neighbors_1[first] += self.neighbors_1[n_g]
                        self.groups_1[n_g] = []
                        self.neighbors_1[n_g] = []
            
            # Marcamos “1” para cada agujero que persista a este nivel (excluye bordes)
            for i_g, group in enumerate(self.groups_1):
                if len(group) > 0 and len([px for px in group if px in self.limits]) == 0:
                    lifetime_1[i_g][level] = 1
        
        # Construimos lifetime_1_extend como [[birth, death], ...]
        self.lifetime_1_extend = [
            [np.min(np.where(life)[0]), np.max(np.where(life)[0]) + 1]
            for life in lifetime_1 if np.sum(life) > 0
        ]
        if len(self.lifetime_1_extend) > 1:
            self.lifetime_1 = list(np.diff(self.lifetime_1_extend, axis=1).flatten())

    def connected_pixels(self, pixels, class_type=1):
        """
        Encuentra grupos de píxeles conectados.
        - class_type = 0 → usa vecinos_0_pixels (8-conexos, H₀).
        - class_type = 1 → usa vecinos_1_pixels (4-conexos, H₁).
        Devuelve (lista_de_grupos, lista_de_vecinos_externos).
        """
        pixels = [int(p) for p in pixels]
        neighbors = [
            self.neighbors_0_pixels[p] if class_type == 0 else self.neighbors_1_pixels[p]
            for p in pixels
        ]
        groups = []
        groups_neighbors = []
        while pixels:
            group = [pixels.pop(0)]
            neighbor = neighbors.pop(0)
            i = 0
            while i < len(pixels):
                if pixels[i] in neighbor:
                    group.append(pixels.pop(i))
                    neighbor += neighbors.pop(i)
                else:
                    i += 1
            groups.append(group)
            groups_neighbors.append(list(set(neighbor)))
        return groups, groups_neighbors

# --------- Clase ImagenTopo ---------
class ImagenTopo:
    """
    Clase envoltura que:
    1. Carga y preprocesa la imagen (pone NaN a 255).
    2. Crea un objeto Topology para esa imagen.
    3. Permite graficar los diagramas/barras de H₀ y H₁.
    """
    def __init__(self, path):
        self.image = load_and_preprocess_image(path)
        self.topology = Topology(self.image)

    def plot_persistence_H0(self):
        """
        Grafica diagrama de persistencia y código de barras para H₀.
        """
        lif0 = np.array(self.topology.lifetime_0_extend)
        if lif0.size == 0:
            print("No se encontraron componentes H₀.")
            return
        # Diagrama de persistencia H₀
        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(lif0[:, 0], lif0[:, 1], color='darkcyan')
        plt.plot([0, self.topology.max_val], [0, self.topology.max_val], 'r--')
        plt.title("Diagrama de Persistencia H₀")
        plt.xlabel("Nacimiento")
        plt.ylabel("Muerte")
        plt.grid(True)
        plt.show()

        # Código de barras H₀
        plt.figure(figsize=(6, 4), dpi=200)
        for idx, (b, d) in enumerate(lif0):
            plt.plot([b, d], [idx, idx], color='darkblue', linewidth=2)
        plt.title("Código de Barras H₀")
        plt.xlabel("Nivel de Intensidad")
        plt.ylabel("Grupo H₀")
        plt.grid(True)
        plt.show()

    def plot_persistence_H1(self):
        """
        Grafica diagrama de persistencia y código de barras para H₁.
        """
        lif1 = np.array(self.topology.lifetime_1_extend)
        if lif1.size == 0:
            print("No se encontraron agujeros H₁.")
            return
        # Diagrama de persistencia H₁
        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(lif1[:, 0], lif1[:, 1], color='tomato')
        plt.plot([0, self.topology.max_val], [0, self.topology.max_val], 'r--')
        plt.title("Diagrama de Persistencia H₁")
        plt.xlabel("Nacimiento")
        plt.ylabel("Muerte")
        plt.grid(True)
        plt.show()

        # Código de barras H₁
        plt.figure(figsize=(6, 4), dpi=200)
        for idx, (b, d) in enumerate(lif1):
            plt.plot([b, d], [idx, idx], color='orangered', linewidth=2)
        plt.title("Código de Barras H₁")
        plt.xlabel("Nivel de Intensidad")
        plt.ylabel("Grupo H₁")
        plt.grid(True)
        plt.show()
    def get_persistence_data(self):
        """
        Devuelve:
            - listas de intervalos extendidos (para diagramas)
            - listas de longitudes de vida (para código de barras)
            - entropía de persistencia para H0 y H1
        """
        lif0_ext = self.topology.lifetime_0_extend
        lif1_ext = self.topology.lifetime_1_extend
        lif0 = self.topology.lifetime_0
        lif1 = self.topology.lifetime_1

        entropy_0 = persistence_entropy(lif0)
        entropy_1 = persistence_entropy(lif1)

        return {
            "H0": {
                "intervals": lif0_ext,
                "lifetimes": lif0,
                "entropy": entropy_0
            },
            "H1": {
                "intervals": lif1_ext,
                "lifetimes": lif1,
                "entropy": entropy_1
            }
        }

# =============================================================================
# Ejecucion principal
# =============================================================================
# if __name__ == "__main__":

#     image_path = r"C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Experimento\130036, 20.11.12, LMLO.png"
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"No se encontró la imagen '{image_path}'")

#     img_topo = ImagenTopo(image_path)
#     data = img_topo.get_persistence_data()

#     # Desempaquetamos la tupla (entropía_pura, entropía_norm) que devolvió get_persistence_data()
#     entropy_H0, norm_H0 = data["H0"]["entropy"]
#     entropy_H1, norm_H1 = data["H1"]["entropy"]

#     # Ahora sí podemos formatear la parte "pura"
#     print(f"Entropía de Persistencia H₀  {entropy_H0:.4f}")
#     print(f"Entropía de Persistencia H₁  {entropy_H1:.4f}")

#     # Y opcionalmente mostrar la normalizada:
#     print(f"Entropía de Shannon normalizada H₀: {norm_H0:.4f}")
#     print(f"Entropía de Shannon normalizada H₁: {norm_H1:.4f}")

#     img_topo.plot_persistence_H0()
#     img_topo.plot_persistence_H1()

def procesar_imagen_topologica(image, target_size=(300, 300)):

    topo = Topology(image)

    # Obtener datos
    lif0_ext = topo.lifetime_0_extend
    lif1_ext = topo.lifetime_1_extend
    lif0 = topo.lifetime_0
    lif1 = topo.lifetime_1

    # Entropías
    pe_h0, sh_h0 = persistence_entropy(lif0)
    pe_h1, sh_h1 = persistence_entropy(lif1)

    # Desempaquetar intervalos
    nac_h0 = [b for b, _ in lif0_ext]
    mur_h0 = [d for _, d in lif0_ext]
    nac_h1 = [b for b, _ in lif1_ext]
    mur_h1 = [d for _, d in lif1_ext]

    # Empaquetar en tupla
    return nac_h0, mur_h0, lif0, pe_h0, sh_h0,nac_h1, mur_h1, lif1, pe_h1, sh_h1

# ruta = r"C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Enfermos\130119, 19.12.12, LCC.png"
# resultados = procesar_imagen_topologica(ruta)

