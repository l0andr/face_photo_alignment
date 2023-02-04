import numpy as np
from numpy.linalg import inv, norm, lstsq, solve
from numpy.linalg import matrix_rank as rank

# from utils.funcs import *


condlimit = 1e9


def cp2tform(uv, xy, method = 'similarity', **kwargs):
    # Infer spatial transformation from control point pairs.
    #   Adapted from CP2TFORM

    # initialize deviation matrices
    xy_dev = []
    uv_dev = []
    options = {'order': 3, "K": []}

    # Assign function according to method and
    # set K = number of control point pairs needed.

    if method == 'similarity':
        findT_fcn = findSimilarity
        options["K"] = 3
    elif method == 'nonreflective similarity':
        findT_fcn = findNonreflectiveSimilarity
        options["K"] = 2
    else:
        raise RuntimeError(f'cp2tform method {method} did not implement')

    # error if user enters too few control point pairs
    M = uv.shape[0]
    if "K" in options and M < options["K"]:
        raise RuntimeError(f"cp2tform: rankError - {options['K']}: {method} - too few control points")

    # get offsets to apply to before/after spatial transformation
    uvShift = getShift(uv)
    xyShift = getShift(xy)
    needToShift = np.any(np.hstack((uvShift, xyShift)) != 0)

    if not needToShift:
        # infer transform
        [trans, output] = findT_fcn(uv, xy)
    else:
        # infer transform for shifted data
        [tshifted, output] = findT_fcn(applyShift(uv, uvShift),
                                       applyShift(xy, xyShift))

        # construct custom tform with tshifted between forward and inverse shifts
        tdata = {'uvShift': uvShift, 'xyShift': xyShift, 'tshifted': tshifted}
        trans = maketform('custom', 2, 2, fwd, inverse, tdata)

    return trans, uv, xy, uv_dev, xy_dev


def findNonreflectiveSimilarity(uv, xy):
    '''
        Find Non-reflective Similarity Transform Matrix 'trans':
    '''
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        if X.shape[0] == X.shape[1]:
            r = solve(X, U)
        else:
            r, _, _, _ = lstsq(X, U, rcond=None)
            r = np.squeeze(r)
    else:
        raise RuntimeError('Error cp2tform two identical points')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ], dtype=float)

    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    trans = maketform('affine', T)
    output = {}

    return trans, output


def findSimilarity(uv, xy):
    '''
        Find Reflective Similarity Transform Matrix 'trans':
    '''

    options = {'K': 3}

    # Solve for trans1
    trans1, trans1_out = findNonreflectiveSimilarity(uv, xy)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_out = findNonreflectiveSimilarity(uv, xyR)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = maketform('affine', trans2r['tdata']['T'] @ TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd_x(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd_x(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_out
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2r_out


def tformfwd_x(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))

    xy = np.dot(uv, trans['tdata']['T'])
    xy = xy[:, 0:-1]
    return xy


def tforminv_x(trans, uv):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd_x(Tinv, uv)
    return xy


def getShift(points):
    tol = 1e+3
    minPoints = np.min(points, axis=0)
    maxPoints = np.max(points, axis=0)
    center = (minPoints + maxPoints) / 2
    span = maxPoints - minPoints
    if (span[0] > 0 and (abs(center[0]) / span[0] > tol)) or (span[1] > 0 and (abs(center[1]) / span[1] > tol)):
        shift = center
    else:
        shift = [0, 0]
    return shift


def applyShift(points, shift):
    shiftedPoints = points - shift
    return shiftedPoints


def undoShift(shiftedPoints, shift):
    points = shiftedPoints + shift
    return points


# -------------------------------
def fwd(u, t):
    x = undoShift(tformfwd_x(applyShift(u, t["tdata"]["uvShift"]), t["tdata"]["tshifted"]), t["tdata"]["xyShift"])
    return x


# -------------------------------
def inverse(x, t):
    u = undoShift(tforminv_x(applyShift(x, t["tdata"]["xyShift"]), t["tdata"]["tshifted"]), t["tdata"]["uvShift"])
    return u


def maketform(*args):
    # MAKETFORM Create spatial transformation structure (TFORM).

    if len(args) <= 0:
        raise RuntimeError(" No input arguments provided for maketform().")

    transform_type = args[0]

    if transform_type == 'affine':
        fcn = affine
    else:
        raise RuntimeError('Maketform:unknownTransformType' + str(args[0]))

    t = fcn(args[1:])

    return t


# --------------------------------------------------------------------------
def assigntform(ndims_in, ndims_out, forward_fcn, inverse_fcn, tdata):
    # Use this function to ensure consistency in the way we assign
    # the fields of each TFORM struct.
    t = {}
    t["ndims_in"] = ndims_in
    t["ndims_out"] = ndims_out
    t["forward_fcn"] = forward_fcn
    t["inverse_fcn"] = inverse_fcn
    t["tdata"] = tdata
    return t


# -------------------------------------------------------------------------
def affine(args):
    # Build an affine TFORM struct.

    if len(args) <= 0 or len(args) > 2:
        raise RuntimeError("[Error]: Invalid number of arguments provided for affine().")

    if len(args) == 2:
        # Construct a 3-by-3 2-D affine transformation matrix A
        # that maps the three points in X to the three points in U.
        U = args[0]
        X = args[1]
        A = construct_matrix(U, X, 'affine')
        A[:, 2] = [0, 0, 1]  # Clean up noise before validating A.
    else:
        A = args[0]

    A = validate_matrix(A, 'affine')

    N = A.shape[1] - 1
    tdata = {}
    tdata['T'] = A
    tdata['Tinv'] = np.linalg.inv(A)

    # In case of numerical noise, coerce the inverse into the proper form.
    tdata['Tinv'][:-1, -1] = 0
    tdata['Tinv'][-1, -1] = 1

    t = assigntform(N, N, fwd_affine, inv_affine, tdata)
    return t


def inv_affine(X, t):
    # INVERSE affine transformation
    #
    # T is an affine transformation structure. X is the row vector to
    # be transformed, or a matrix with a vector in each row.

    U = trans_affine(X, t, 'inverse')
    return U


# --------------------------------------------------------------------------
def fwd_affine(U, t):
    # FORWARD affine transformation
    #
    # T is an affine transformation structure. U is the row vector to
    # be transformed, or a matrix with a vector in each row.

    X = trans_affine(U, t, 'forward')
    return X


# --------------------------------------------------------------------------
def trans_affine(X, t, direction):
    # Forward/inverse affine transformation method
    #
    # T is an affine transformation structure. X is the row vector to
    # be transformed, or a matrix with a vector in each row.
    # DIRECTION is either 'forward' or 'inverse'.

    if direction == 'forward':
        M = t["tdata"]["T"]
    elif direction == 'inverse':
        M = t["tdata"]["Tinv"]
    else:
        raise RuntimeError('[Error]:maketform:invalidDirection')

    X1 = np.hstack([X, np.ones((X.shape[0], 1))])  # Convert X to homogeneous coordinates
    U1 = X1 @ M  # Transform in homogeneous coordinates
    U = U1[:, :-1]  # Convert homogeneous coordinates to U
    return U


# ---------------------------------------------------------------
def construct_matrix(U, X, transform_type):
    # Construct a 3-by-3 2-D transformation matrix A
    # that maps the points in U to the points in X.

    if transform_type == 'affine':
        nPoints = 3
        unitFcn = UnitToTriangle
    else:
        raise RuntimeError('construct_matrix: incorrect transform_type')

    if (U.shape[0] not in [nPoints, 2]) or (U.shape[1] not in [nPoints, 2]):
        raise RuntimeError('images:maketform:invalidUSize' + str(nPoints))

    if (X.shape[0] not in [nPoints, 2]) or (X.shape[1] not in [nPoints, 2]):
        raise RuntimeError('images:maketform:invalidXSize' + str(nPoints))

    Au = unitFcn(U)
    if np.linalg.cond(Au) > condlimit:
        print('[Warning]: images:maketform:conditionNumberOfUIsHigh')

    Ax = unitFcn(X)
    if np.linalg.cond(Ax) > condlimit:
        print('[Warning]: images:maketform:conditionNumberOfXIsHigh')

    # (unit shape) * Au = U
    # (unit shape) * Ax = X
    #
    # U * inv(Au) * Ax = (unit shape) * Ax = X and U * A = X,
    # so inv(Au) * Ax = A, or Au * A = Ax, or A = Au \ Ax.

    # A = Au \ Ax
    if Au.shape[0] == Au.shape[1]:
        A = np.linalg.solve(Au, Ax)
    else:
        A = np.linalg.lstsq(Au, Ax)

    if np.any(np.isinf(A)):
        RuntimeError('[Error]: images:maketform:collinearPointsinUOrX')

    A = A / A[-1, -1]
    return A


# ---------------------------------------------------------------

def validate_matrix(A, transform_type):
    # Make sure A is finite.
    if np.any(np.isinf(A)):
        RuntimeError('images:maketform:aContainsInfs')

    # Make sure A is (N + 1)-by-(N + 1).  Append a column if needed for 'affine'.
    N = A.shape[0] - 1
    if transform_type == 'affine' and A.shape[1] == N:
        A[:, N] = np.append(np.zeros((N, 1)), [1])

    if N < 1 or (A.shape[1] != N + 1):
        RuntimeError('images:maketform:invalidASize')

    if transform_type == 'affine':
        # Validate the final column of A.
        if np.any(np.not_equal(A[:, N], np.append(np.zeros((N, 1)), [1]))):
            RuntimeError('images:maketform:invalidAForAffine')
    elif transform_type == 'projective':
        # Validate lower right corner of A
        if abs(A[N, N]) <= 100 * np.finfo(float).eps * np.linalg.norm(A, 2):
            print('[Warning]:images:maketform:lastElementInANearZero')

    if np.linalg.cond(A) > condlimit:
        print('[Warning]: images:maketform:conditionNumberofAIsHigh', np.linalg.cond(A))

    return A


def UnitToTriangle(X):
    # Computes the 3-by-3 two-dimensional affine transformation
    # matrix A that maps the unit triangle ([0 0], [1 0], [0 1])
    # to a triangle with corners (X(1,:), X(2,:), X(3,:)).
    # X must be 3-by-2, real-valued, and contain three distinct
    # and non-collinear points. A is a 3-by-3, real-valued matrix.

    A = np.array([[X[1, 0] - X[0, 0], X[1, 1] - X[0, 1], 0],
                  [X[2, 0] - X[0, 0], X[2, 1] - X[0, 1], 0],
                  [X[0, 0], X[0, 1], 1]])
    return A
# ---------------------------------------------------------------
