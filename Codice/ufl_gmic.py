import time, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import gurobipy as gp
from gurobipy import GRB

def solve_ufl_with_gomory_gmi(f, c, max_cuts=10000, time_limit=900, tol_improvement=0.01):

    """
    Risolve il problema UFL usando i Gomory Mixed-Integer Cuts.
    
    Parametri:
    f: array dei costi fissi di apertura delle facilities (lunghezza m)
    c: matrice (lista di liste) dei costi di assegnamento (n righe x m colonne)
    max_cuts: limite massimo di tagli
    time_limit: in secondi (15 minuti = 900s)
    tol_improvement: 1% di miglioramento minimo

    """
    
    n_clienti = len(c)      
    m_facilities = len(f)         
    
    model = gp.Model("UFL_Relaxed_Model_GMI")
    model.setParam('OutputFlag', 0) 

    # Disabilitazione Presolve e uso Dual Simplex
    model.setParam('Presolve', 0)
    model.setParam('Method', 1)
    
    # Creazione variabili (continue per il rilassamento)
    # Senza ub=1.0, le variabili fuori base saranno impostate a 0, 
    # altrimenti Gurobi le imposta all'upper bound sballando i tagli GMI
    y = model.addVars(m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
    x = model.addVars(n_clienti, m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    
    # Creazione delle variabili Slack
    # Nella formulazione debole serve una sola slack per ogni facility
    s = model.addVars(m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
    
    # Funzione Obiettivo
    obj_setup = gp.quicksum(f[j] * y[j] for j in range(m_facilities))
    obj_routing = gp.quicksum(c[i][j] * x[i, j] for i in range(n_clienti) for j in range(m_facilities))
    model.setObjective(obj_setup + obj_routing, GRB.MINIMIZE)
    
    # Vincoli di Domanda
    for i in range(n_clienti):
        model.addConstr(gp.quicksum(
            x[i, j] for j in range(m_facilities)) == 1, 
            name=f"domanda_{i}"
        )
        
    # Vincoli di Attivazione in Formulazione Debole
    # Somma(x_ij) - N*y_j + s_j == 0
    for j in range(m_facilities):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n_clienti)) - (n_clienti * y[j]) + s[j] == 0, 
            name=f"attivazione_debole_{j}"
        )

    print("\n")
    print("Risoluzione del Rilassamento Lineare iniziale...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        bound_iniziale = model.ObjVal   
        print(f"Lower Bound iniziale: {bound_iniziale:.2f}")
    else:
        print("Errore nella risoluzione del modello base.")
        return None

    start_time = time.time()
    n_tagli_inseriti = 0
    ultimo_bound = model.ObjVal
    
    print("\nAvvio del ciclo del piano di taglio (GMI)...")
    print(f"{'Iterazione':<12} | {'Lower Bound':<15} | {'Miglioramento':<15} | {'Tagli Totali':<15}")
    print("-" * 70)
    
    iterazione = 1
    continua_loop = True

    while continua_loop:
        tempo_trascorso = time.time() - start_time
        if tempo_trascorso > time_limit:
            break
        if n_tagli_inseriti >= max_cuts:
            break
        
        if iterazione > 1:    
            model.optimize()

        if model.status != GRB.OPTIMAL:
            print("\n[Errore] Il modello è diventato inammissibile.")
            break
            
        nuovo_bound = model.ObjVal
        
        miglioramento_pct = (nuovo_bound - ultimo_bound) / abs(ultimo_bound) if ultimo_bound != 0 else 0
        miglioramento_str = f"{miglioramento_pct * 100:.2f}%"
            
        print(f"{iterazione:<12} | {nuovo_bound:<15.4f} | {miglioramento_str:<15} | {n_tagli_inseriti:<15}")
        
        if iterazione > 1 and miglioramento_pct < tol_improvement:
            print(f"\n[Terminazione] Il miglioramento ({miglioramento_str}) è inferiore a {tol_improvement*100}%.")
            break
            
        ultimo_bound = nuovo_bound
        all_vars = model.getVars() 

        # Estrazione dello stato della base da Gurobi
        vbasis = np.array(model.getAttr("VBasis", all_vars))
        
        # Estrazione della matrice sparsa dei coefficienti (A) e costruzione della matrice di Base (B)
        A = model.getA()
        base_var_idx = np.where(vbasis == 0)[0]
        B_matrix = A[:, base_var_idx].tocsc()
        
        variabile_frazionaria_trovata = False

        # Cerchiamo la prima variabile frazionaria (solo tra x e y, non ci interessano le slack frazionarie)
        for idx, v in enumerate(all_vars):
            # Saltiamo le variabili di slack (iniziano per 's') e quelle artificiali di Gurobi
            if not (v.VarName.startswith('x') or v.VarName.startswith('y')):
                continue
                
            valore = v.X
            parte_intera = int(round(valore))
            
            # Verifichiamo che la variabile sia in base (vbasis == 0) e sia frazionaria
            if vbasis[idx] == 0 and abs(valore - parte_intera) > 1e-4:
                variabile_frazionaria_trovata = True
                
                # Calcolo esplicito della riga del tableau
                # Risolviamo y * B = e_k  ---> y = B^-1 * e_k
                pos_in_base = np.where(base_var_idx == idx)[0][0]
                e_k = np.zeros(B_matrix.shape[0])
                e_k[pos_in_base] = 1.0
                
                y_vec = spsolve(B_matrix.T, e_k)
                bar_a = y_vec @ A # Vettore dei coefficienti aggiornati per tutte le variabili
                
                # Calcolo f_0 della variabile in base
                f_0 = valore - math.floor(valore)
                cut_expr = gp.LinExpr()

                for j, j_var in enumerate(all_vars):
                    if vbasis[j] != 0: # Calcoliamo i coefficienti GMI solo per le variabili fuori base
                        bar_a_j = bar_a[j]
                        
                        # Filtriamo errori numerici (coefficienti nulli)
                        if abs(bar_a_j) < 1e-6:
                            continue
                            
                        # Controllo: La variabile è Intera o Continua?
                        is_integer = j_var.VarName.startswith('x') or j_var.VarName.startswith('y')
                        
                        if is_integer:
                            # --- Formula GMI per var intere
                            f_j = bar_a_j - math.floor(bar_a_j)
                            if f_j <= f_0:
                                coeff = f_j / f_0
                            else:
                                coeff = (1.0 - f_j) / (1.0 - f_0)
                        else:
                            # --- Formula GMI per var continue
                            if bar_a_j >= 0:
                                coeff = bar_a_j / f_0
                            else:
                                coeff = -bar_a_j / (1.0 - f_0)
                                
                        # Aggiungiamo il termine al taglio se il coefficiente è significativo
                        if abs(coeff) > 1e-6:
                            cut_expr.addTerms(coeff, j_var)
            
                # Variabile di surplus esplicita per il taglio
                s_cut = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"s_cut_{n_tagli_inseriti}")
                
                # Inseriamo il taglio come UGUAGLIANZA (cut_expr - s_cut == 1.0)
                model.addConstr(cut_expr - s_cut == 1.0, name=f"GMI_{n_tagli_inseriti}")
                n_tagli_inseriti += 1
                    
                break # Generiamo un taglio per volta

        if not variabile_frazionaria_trovata:
            print("\n[Ottimo Trovato] Tutte le variabili decisionali sono intere!")
            break
            
        iterazione += 1
        
    print(f"\nProcesso terminato. Bound finale: {model.ObjVal:.4f}")
    return model, bound_iniziale, n_tagli_inseriti


