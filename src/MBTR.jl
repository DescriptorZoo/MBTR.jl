module MBTR

using ASE
using JuLIP
using PyCall

function __init__()
    py"""
    import qmmlpack as qmml
    
    import math
    import numpy as np
    import itertools

    from ase import Atoms
    from ase.data import atomic_numbers, chemical_symbols
    
    PI = 3.14159265359
    unit_BOHR = 0.52917721067
    
    # DEFINITONS OF QMML-PACK SUPPORT FUNCTIONS
    #
    def basis_check(cell):
        # Checking lattice vectors whether they all are zeros.
        # If x direction is all zero values, 
        # we supply 1,0,0 vector instead.
        # This applies all other directions
        return True if( np.linalg.norm(cell[0]) < 1E-12 and 
                        np.linalg.norm(cell[1]) < 1E-12 and 
                        np.linalg.norm(cell[2]) < 1E-12 ) else False

    def qmml_mbtr_calc(k, z, r, elms, basis=None, flatten=True):
        repr_ = []
        for t in k:
            accuracy = t[2] # accuracy of parameter expansion
            margins = t[1]  # [1/dmax, 1/dmin] plus margin for t[0][0]==2
                            # [0,Pi] plus margin for t[0][0]==3
            defs = t[0] # Parameters for representation
            (dsigma,) = t[3] # theta parameter for this representation
            definitions = (defs[0], 
                           defs[1], 
                           defs[2], 
                           (defs[3][0], (dsigma,)), 
                           defs[4], 
                           defs[5], 
                           defs[6])
            repr_.append(qmml.many_body_tensor(z, r, margins, definitions,
                         basis=basis, acc=accuracy, elems=elms, flatten=flatten))
        return repr_

    def mbtr_reprf(k, atoms, kernelf=None, kernel_theta=None, flatten=True):
        #MBTR, k=2, inverse distances, quadratic weighting
        #      k=3, angles
        if isinstance(atoms, Atoms):
            ase_atoms = [atoms]
            all_z = atoms.numbers
            all_list = all_z
        else:
            ase_atoms = atoms
            all_z = np.asarray([a.numbers for a in atoms])
            all_list = list(itertools.chain.from_iterable(all_z))
        elms = np.unique(all_list)
        z_basis = []
        r_basis = []
        basis = []
        basis_ids = []
        z_no_basis = []
        r_no_basis = []
        no_basis_ids = []
        # Split basis to None and Atoms.cell
        for ai, a in enumerate(ase_atoms):
            if hasattr(a, 'cell'):
                # check if cell only includes zeros
                if not basis_check(a.cell):
                    z_basis.append(a.numbers)
                    r_basis.append(a.positions)
                    basis.append(a.cell)
                    basis_ids.append(ai)
                else:
                    z_no_basis.append(a.numbers)
                    r_no_basis.append(a.positions)
                    no_basis_ids.append(ai)
            else:
                z_no_basis.append(a.numbers)
                r_no_basis.append(a.positions)
                no_basis_ids.append(ai)

        # representation
        repr_ = []
        all_ids = {}
        if z_basis:
            for i, bi in enumerate(basis_ids):
                all_ids[bi] = i 
            repr_.extend(qmml_mbtr_calc(k, z_basis, r_basis, 
                                        elms, basis=basis, 
                                        flatten=flatten))
        if z_no_basis:
            for i, bi in enumerate(no_basis_ids):
                all_ids[bi] = i + len(basis_ids)
            repr_.extend(qmml_mbtr_calc(k, z_no_basis, r_no_basis, 
                                        elms, basis=None, 
                                        flatten=flatten))
        repr_ = np.concatenate(repr_, axis=1)
        M = None
        if kernelf is not None:
            # kernel matrix
            M = kernelf(repr_, theta=kernel_theta)
        return repr_, M, all_ids

    def paramf(aa, bb, cc):
        # Parameter splitting function dividing 
        # parameters into three blocks of sizes a, b, c.
        #
        # a, b, c are numbers of parameters.
        # a can also be a list of numbers of parameters.
        #
        # paramf([1,2], 0, 1)(['a', 'b', 'c', 'd']) 
        #    -> 
        # ([('a',), ('b', 'c')], tuple(), ('d',))
        def f(a, b, c, *theta):
            if qmml.is_sequence(a):
                # multiple representation functions
                # as we have only need for cases 2 
                # and 3, no need to generalize
                assert len(a) == 2 or len(a) == 3
                resa = [ tuple(theta[:a[0]]), 
                         tuple(theta[a[0]:a[0]+a[1]]) ]
                if len(a) == 3: resa.append( 
                                tuple(theta[a[0]+a[1]:a[0]+a[1]+a[2]]) )
                a = np.sum(a)
            else:
                # single representation function
                resa = tuple(theta[:a])
            return ( resa, tuple(theta[a:a+b]), tuple(theta[a+b:a+b+c]) )
        return lambda *theta: f(aa, bb, cc, *theta)
    
    def calc_desc(atoms_list, desc, reprfs, kfunc=None, 
                  theta_kernel=None, flatten=True):
        out_desc = []
        # Split dataset into non-PBC and PBC 
        # structures and calculate descriptors
        pbc_structs = []
        nonpbc_structs = []
        for aseat in atoms_list:
            if (aseat.pbc[0] == False and 
                aseat.pbc[1] == False and 
                aseat.pbc[2] == False):
                nonpbc_structs.append(aseat)
            else:
                pbc_structs.append(aseat)
        if pbc_structs:
            # Calculate descriptor for structures with pbc
            rtn_desc, kern, ids = desc(reprfs, pbc_structs, kernelf=kfunc, 
                                       kernel_theta=theta_kernel, 
                                       flatten=flatten)
            for count, ase_at in enumerate(pbc_structs):
                # Setting database 
                out_desc.append({
                    'representation' : rtn_desc[ids[count]],
                    'kernel'         : kern[ids[count]],
                    'symbols'        : ase_at.get_chemical_symbols()
                    })
        if nonpbc_structs:
            # Calculate descriptor for structures with pbc
            rtn_desc, kern, ids = desc(reprfs, nonpbc_structs, kernelf=kfunc, 
                                       kernel_theta=theta_kernel, 
                                       flatten=flatten)
            for count, ase_at in enumerate(nonpbc_structs):
                # Setting database 
                out_desc.append({
                    'representation' : rtn_desc[ids[count]],
                    'kernel'         : kern[ids[count]],
                    'symbols'        : ase_at.get_chemical_symbols()
                    })
        return out_desc


    def MBTR_desc(at, reprs=2, feature='r', kernelf='gaussian', 
                  accuracy=0.001, dist_min=0.1, dist_max=1.1, 
                  dist_step=100, ang_min=-0.15, ang_max=3.45575, 
                  ang_step=100, dist_width=-4.0, ang_width=-3.0, 
                  sigma=0.0, stepsize=0.5, reg_strength=-20.0, 
                  reg_direction=-1, cutoff=6.0):
        # at            : ASE Atoms
        # reprs         : 0=only distances, 1=only angles, 2=both.
        # feature       : kernel='k' or representation='r'
        # kernelf       : 'gaussian', 'linear', or 'laplacian'.
        # accuracy      : Accuracy for MBTR
        # dist_min      : Minimum distance
        # dist_max      : Maximum distance
        # dist_step     : Number of distance steps
        # ang_min       : Minimum for angle margins
        # ang_max       : Maximum for angle margins
        # ang_step      : Number of angle margin step
        # dist_width    : 2-body MBTR normal distribution width.
        # ang_width     : 3-body MBTR normal distribution width. 
        # sigma         : kernel basis sigma.
        # stepsize      : Stepsize for MBTR variables.
        # reg_strength  : Regularization strength.
        # reg_direction : Regularization direction.
        # cutoff        : Reserved if needed. Currently, not used!
    
        if 'laplacian' in kernelf:
            kfunc = qmml.kernel_laplacian
        elif 'linear' in kernelf:
            kfunc = qmml.kernel_linear
        else:
            kfunc = qmml.kernel_gaussian

        dist_repr = [
            (2, '1/distance', 'identity^2', 
                ('normal', 0.), 'identity', 
                'noreversals', 'noreversals'
            ),
            (float(dist_min), 
             float(dist_max/int(dist_step)), 
             int(dist_step)),
            float(accuracy),
            0.
            ]
        ang_repr = [
            (3, 'angle', '1/dotdotdot', 
                ('normal', 0.), 'identity', 
                'noreversals', 'noreversals'
            ),
            (float(ang_min), 
             float(ang_max/int(ang_step)), 
             int(ang_step)),
            float(accuracy),
            0.
            ]
        reprfs = []
        if reprs > 1:
            reprfs.append(dist_repr)
        if reprs > 0:
            reprfs.append(ang_repr)
        if reprs < 1:
            reprfs.append(dist_repr)

        variables = (
            { 'value': dist_width, 'priority': 2, 'stepsize': stepsize },
            { 'value': ang_width, 'priority': 1, 'stepsize': stepsize }, 
            { 'value': sigma, 'priority': 3, 'stepsize': stepsize },
            { 'value': reg_strength, 'priority': 4, 
              'direction': reg_direction, 'minimum': reg_strength } 
            )
        parmf = paramf([1,1],1,1)
        theta = (dist_width, ang_width, sigma, reg_strength)
        (theta_repr, theta_kernel, theta_method) = parmf(
                                                    *(2.**np.asarray(theta)))
        for ki, k in enumerate(reprfs):
            reprfs[ki][3] = theta_repr[ki]

        fingerprints = calc_desc([at], mbtr_reprf, 
                                 reprfs, kfunc, 
                                 theta_kernel)
        if feature.startswith('r'):
            fs = np.array(fingerprints[0]['representation'])
        else:
            fs = np.array(fingerprints[0]['kernel'])
        return fs
    """
end


export mbtr

# Please check parameter definitions in python function above.
function mbtr(at; reprs=2, feature="r", kernelf="gaussian", 
              accuracy=0.001, dist_min=0.1, dist_max=1.1, 
              dist_step=100, ang_min=-0.15, ang_max=3.45575, 
              ang_step=100, dist_width=-4.0, ang_width=-3.0, 
              sigma=0.0, stepsize=0.5, reg_strength=-20.0, 
              reg_direction=-1, cutoff=6.0)
    atom_struct = ASEAtoms(at)
    # Calculate descriptor
    mbtr_rtn = py"MBTR_desc"(atom_struct.po, reprs=reprs, 
                             feature=feature, kernelf=kernelf, 
                             accuracy=accuracy, dist_min=dist_min, 
                             dist_max=dist_max, dist_step=dist_step,
                             ang_min=ang_min, ang_max=ang_max, 
                             ang_step=ang_step, dist_width=dist_width, 
                             ang_width=ang_width, sigma=sigma, 
                             stepsize=stepsize, reg_strength=reg_strength, 
                             reg_direction=reg_direction, cutoff=cutoff)
    return mbtr_rtn
end

end # module
