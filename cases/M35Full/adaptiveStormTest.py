# ---------------------------------------------------------
# Pomocné funkce simulující GPU operace
# ---------------------------------------------------------

def inclusive_max_scan(arr):
    """Simuluje GPU forward inclusive max-scan."""
    result = [0] * len(arr)
    current_max = arr[0]
    for i in range(len(arr)):
        if arr[i] > current_max:
            current_max = arr[i]
        result[i] = current_max
    return result

# ---------------------------------------------------------
# 1. Inicializace "zhuštěných" vstupních dat (N = 15)
# Data musí být seřazena K (pomalé) -> J -> I (rychlé)
# ---------------------------------------------------------
# Z=0 (K=0): 3 řádky (J=0, 1, 2)
# Z=1 (K=1): 2 řádky (J=0, 1)

K_coarse = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
J_coarse = [0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2]
I_coarse = [0, 2, 5, 1, 3, 0, 2, 0, 1, 4, 1, 3, 6, 2, 5]

N_refined = len(K_coarse)

print(f"Počet hrubých buněk (N_refined): {N_refined}")
print(f"Očekávaný počet jemných buněk: {N_refined * 8}\n")

# ---------------------------------------------------------
# 2. Vytvoření pomocných polí (krok 6.4)
# ---------------------------------------------------------

# --- firstInPlane ---
firstInPlane_flags = [0] * N_refined
for c in range(N_refined):
    if c == 0 or K_coarse[c] != K_coarse[c-1]:
        firstInPlane_flags[c] = c
firstInPlane = inclusive_max_scan(firstInPlane_flags)

# --- lastInPlane (s využitím Mailbox tricku místo reverse-scanu) ---
lastInPlane_flags = [0] * N_refined
for c in range(N_refined):
    if c == N_refined - 1 or K_coarse[c] != K_coarse[c+1]:
        # Zápis konce roviny na index jejího začátku
        lastInPlane_flags[firstInPlane[c]] = c 
lastInPlane = inclusive_max_scan(lastInPlane_flags)

# --- firstInRow ---
firstInRow_flags = [0] * N_refined
for c in range(N_refined):
    # Nový řádek začíná, pokud se změní J, NEBO pokud se změní K (nová rovina)
    if c == 0 or J_coarse[c] != J_coarse[c-1] or K_coarse[c] != K_coarse[c-1]:
        firstInRow_flags[c] = c
firstInRow = inclusive_max_scan(firstInRow_flags)

# --- lastInRow (Mailbox trick) ---
lastInRow_flags = [0] * N_refined
for c in range(N_refined):
    if c == N_refined - 1 or J_coarse[c] != J_coarse[c+1] or K_coarse[c] != K_coarse[c+1]:
        lastInRow_flags[firstInRow[c]] = c
lastInRow = inclusive_max_scan(lastInRow_flags)


print("Kontrola pomocných polí pro prvních 5 buněk:")
for i in range(5):
    print(f"Buňka {i} (I:{I_coarse[i]}, J:{J_coarse[i]}, K:{K_coarse[i]}): "
          f"firstPlane={firstInPlane[i]}, lastPlane={lastInPlane[i]}, "
          f"firstRow={firstInRow[i]}, lastRow={lastInRow[i]}")
print("...")

# ---------------------------------------------------------
# 3. Alokace jemné sítě a výpočet indexů (krok 6.5)
# ---------------------------------------------------------

N_fine = N_refined * 8
K_fine = [0] * N_fine
J_fine = [0] * N_fine
I_fine = [0] * N_fine

# Simulujeme paralelní kernel přes všechny rafinované buňky
for c in range(N_refined):
    for kAdd in range(2):
        for jAdd in range(2):
            for iAdd in range(2):
                
                # ZDE JE TVÁ ROVNICE (s opraveným +1 u jAdd)
                idx = (firstInPlane[c] * 8 
                       + (lastInPlane[c] - firstInPlane[c] + 1) * 4 * kAdd 
                       + (firstInRow[c] - firstInPlane[c]) * 4 
                       + (lastInRow[c] - firstInRow[c] + 1) * 2 * jAdd 
                       + (c - firstInRow[c]) * 2 
                       + iAdd)
                
                # Zápis do jemné sítě
                K_fine[idx] = 2 * K_coarse[c] + kAdd
                J_fine[idx] = 2 * J_coarse[c] + jAdd
                I_fine[idx] = 2 * I_coarse[c] + iAdd

# ---------------------------------------------------------
# 4. Kontrola setřídění jemné sítě
# ---------------------------------------------------------

is_sorted = True
print("\nProvádím kontrolu setřídění jemné sítě (K -> J -> I)...")

for i in range(1, N_fine):
    prev_cell = (K_fine[i-1], J_fine[i-1], I_fine[i-1])
    curr_cell = (K_fine[i], J_fine[i], I_fine[i])
    
    # Python porovnává n-tice přesně tak, jak potřebujeme: první prvek, pak druhý, atd.
    if prev_cell >= curr_cell:
        print(f"[CHYBA] Porušení řazení na indexu {i}: Předchozí {prev_cell} >= Aktuální {curr_cell}")
        is_sorted = False
        break

if is_sorted:
    print("[ÚSPĚCH] Jemná síť je perfektně setříděná!")
    
    print("\nUkázka prvních 16 buněk jemné sítě (děti prvních 2 hrubých buněk z prvního řádku):")
    for i in range(16):
        print(f"Fine Index {i:>3}: K={K_fine[i]}, J={J_fine[i]}, I={I_fine[i]}")
