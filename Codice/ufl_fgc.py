import time, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import gurobipy as gp
from gurobipy import GRB

def solve_ufl_with_fgc(f, c, max_cuts=10000, time_limit=900, tol_improvement=0.01):
    """
    Risolve il problema UFL usando i Fractional Gomory Cuts.
    
    Parametri:
    f: array dei costi fissi di apertura delle facilities (lunghezza m)
    c: matrice (lista di liste) dei costi di assegnamento (n righe x m colonne)
    max_cuts: limite massimo di tagli
    time_limit: in secondi (15 minuti = 900s)
    tol_improvement: 1% di miglioramento minimo

    """
    
    # Dimensioni del problema ricavate dai dati in input
    n_clienti = len(c)      # |I|
    m_facilities = len(f)   # |J|
    
    # Inizializzazione del modello Gurobi
    model = gp.Model("UFL_Relaxed_Model_FGC")
    model.setParam('OutputFlag', 0) 

    model.setParam('Presolve', 0)
    model.setParam('Method', 1) # Dual Simplex
    
    # Creazione delle variabili (Rilassamento Lineare)
    y = model.addVars(m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
    x = model.addVars(n_clienti, m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    s = model.addVars(m_facilities, vtype=GRB.CONTINUOUS, lb=0.0, name="s")

    # Funzione Obiettivo
    # Minimizzare: somma(f_j * y_j) + somma(c_ij * x_ij)
    obj_setup = gp.quicksum(f[j] * y[j] for j in range(m_facilities))
    obj_routing = gp.quicksum(c[i][j] * x[i, j] for i in range(n_clienti) for j in range(m_facilities))
    model.setObjective(obj_setup + obj_routing, GRB.MINIMIZE)
    
    # Vincoli di Domanda (ogni cliente deve essere servito)
    for i in range(n_clienti):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m_facilities)) == 1, 
            name=f"domanda_{i}"
        )
        
    # Vincoli di Attivazione 
    for j in range(m_facilities):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n_clienti)) - (n_clienti * y[j]) + s[j] == 0, 
            name=f"attivazione_debole_{j}"
        )
            
    # Prima ottimizzazione (Rilassamento continuo iniziale)
    print("\n")
    print("Risoluzione del Rilassamento Lineare iniziale (FGC)...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        bound_iniziale = model.ObjVal   
        print(f"Lower Bound iniziale: {bound_iniziale:.2f}")
    else:
        print("Errore nella risoluzione del modello base.")
        return None

    # --- INIZIO DEL CICLO DEL PIANO DI TAGLIO ---

    start_time = time.time()
    n_tagli_inseriti = 0
    ultimo_bound = model.ObjVal
    
    print("\nAvvio del ciclo del piano di taglio (Fractional Cuts)...")
    print(f"{'Iterazione':<12}{'Lower Bound':<15}{'Miglioramento %':<18}{'Tagli Totali':<12}")
    print("-" * 70)
    
    iterazione = 1
    continua_loop = True

    while continua_loop:
        # Controlliamo i criteri di terminazione legati a limiti statici
        tempo_trascorso = time.time() - start_time
        if tempo_trascorso > time_limit:
            print(f"\n[Terminazione] Raggiunto il Time Limit di {time_limit} secondi.")
            break
        if n_tagli_inseriti >= max_cuts:
            print(f"\n[Terminazione] Raggiunto il numero massimo di tagli ({max_cuts}).")
            break
            
        # Ottimizziamo il modello corrente (con i tagli precedentemente aggiunti)
        if iterazione > 1:
            model.optimize()
        
        if model.status != GRB.OPTIMAL:
            print("\n[Errore] Il modello è diventato inammissibile o illimitato.")
            break
            
        nuovo_bound = model.ObjVal
        
        # Calcolo del miglioramento percentuale rispetto all'iterazione precedente
        miglioramento_pct = (nuovo_bound - ultimo_bound) / abs(ultimo_bound) if ultimo_bound != 0 else 0
            
        print(f"{iterazione:<12}{nuovo_bound:<15.4f}{miglioramento_pct*100:<17.2f}%{n_tagli_inseriti:<12}")
        
        # Criterio di terminazione basato sul miglioramento inferiore all'1%
        # Applicato solo dalla seconda iterazione del ciclo
        if iterazione > 1 and miglioramento_pct < tol_improvement:
            print(f"\n[Terminazione] Il miglioramento del bound ({miglioramento_pct*100:.2f}%) è inferiore all'1%.")
            break
            
        ultimo_bound = nuovo_bound
        
        # Identificazione delle variabili frazionarie e generazione del Taglio di Gomory
        all_vars = model.getVars() # Variabili del modello fornite da gurobi
        vbasis = np.array(model.getAttr("VBasis", all_vars))
        A = model.getA()
        base_var_idx = np.where(vbasis == 0)[0]
        B_matrix = A[:, base_var_idx].tocsc()

        variabile_frazionaria_trovata = False

        # Cercha la prima variabile che ha un valore frazionario all'ottimo
        for idx, v in enumerate(all_vars):
            if not (v.VarName.startswith('x') or v.VarName.startswith('y')):
                continue
                
            valore = v.X
            parte_intera = int(round(valore))
            
            if vbasis[idx] == 0 and abs(valore - parte_intera) > 1e-4:
                variabile_frazionaria_trovata = True
                
                pos_in_base = np.where(base_var_idx == idx)[0][0]
                e_k = np.zeros(B_matrix.shape[0])
                e_k[pos_in_base] = 1.0
                
                y_vec = spsolve(B_matrix.T, e_k)
                bar_a = y_vec @ A 
                
                # Calcolo f_0 della variabile in base
                f_0 = valore - math.floor(valore)
                cut_expr = gp.LinExpr()

                for j, j_var in enumerate(all_vars):
                    if vbasis[j] != 0: 
                        bar_a_j = bar_a[j]
                        
                        if abs(bar_a_j) < 1e-6:
                            continue
                            
                        # Matematica FGC  per tutte le variabili
                        f_j = bar_a_j - math.floor(bar_a_j)
                        
                        if abs(f_j) > 1e-6:
                            cut_expr.addTerms(f_j, j_var)

                # Inserimento del taglio esplicito con la variabile di surplus (s_cut)
                # Formula canonica: sum(f_j * x_j) >= f_0  --> sum(f_j * x_j) - s_cut == f_0
                s_cut = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"s_cut_fgc_{n_tagli_inseriti}")
                model.addConstr(cut_expr - s_cut == f_0, name=f"FGC_{n_tagli_inseriti}")
                
                n_tagli_inseriti += 1
                break 

        if not variabile_frazionaria_trovata:
            print("\n[Ottimo Trovato] Tutte le variabili decisionali sono intere!")
            break
            
        iterazione += 1
        
    print(f"\nProcesso terminato. Bound finale: {model.ObjVal:.4f}")
    return model, bound_iniziale, n_tagli_inseriti