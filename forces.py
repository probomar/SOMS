from setup import *


def normal_force():
    normal_vector_femur = np.array(flex.cell_normals)
    point_femur = np.array(femur.cell_centers().points)

    area_femur = flex.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    area_femur = area_femur.reshape([area_femur.shape[0], 1])

    N = np.empty([0, 3])
    stress = np.empty([0, 1])

    for m in range(point_femur.shape[0]):
        pA = point_femur[m] - 10000 * normal_vector_femur[m]
        pB = point_femur[m] + 10000 * normal_vector_femur[m]
        points_tibia = tibia.ray_trace(pA, pB)[0]

        points_femoral_cartilage = flex_cartilage.ray_trace(pA, pB)[0]
        if np.shape(points_femoral_cartilage)[0] >= 2:
            femoral_cartilage_distance = np.linalg.norm(points_femoral_cartilage[0, :] - points_femoral_cartilage[1, :])
        else:
            femoral_cartilage_distance = 0

        points_tibial_cartilage = tibial_cartilage.ray_trace(pA, pB)[0]
        if np.shape(points_tibial_cartilage)[0] >= 2:
            tibial_cartilage_distance = np.linalg.norm(points_tibial_cartilage[0, :] - points_tibial_cartilage[1, :])
        else:
            tibial_cartilage_distance = 0

        cartilage_distance = femoral_cartilage_distance + tibial_cartilage_distance

        distance = np.empty([0])

        if not points_tibia.any():
            stress = np.append(stress, [[0]], axis=0)
            N = np.append(N, [[0, 0, 0]], axis=0)
        else:
            for j in range(points_tibia.shape[0]):
                distance = np.append(distance, np.linalg.norm(point_femur[m, :] - points_tibia[j, :]))

            min_distance = np.min(distance)

            if min_distance < cartilage_distance:
                stress = np.append(stress, [[- E * (min_distance - cartilage_distance) / cartilage_distance]], axis=0)
                N = np.append(N, [E * (min_distance - cartilage_distance) / cartilage_distance * area_femur[m]
                                  * normal_vector_femur[m]], axis=0)
            else:
                stress = np.append(stress, [[0]], axis=0)
                N = np.append(N, [[0, 0, 0]], axis=0)

    N_forces = N.reshape([N.shape[0], 3])

    N = N_forces[0]
    soaN = point_femur[0]

    for j in range(N_forces.shape[0] - 1):
        N, MN, soaN = result_of_forces_and_moments(N, soaN, N_forces[j + 1], point_femur[j + 1])

    return N, soaN, stress


def moment_of_force(force, site_of_act, point=np.array([0, 0, 0])):
    if force.all() == 0:
        moment = np.array([0, 0, 0])
    else:
        param = (point - site_of_act @ force) / (force @ force)
        arm = site_of_act + force * param - point
        moment = np.array([- force[1] * arm[2] + force[2] * arm[1],
                           - force[2] * arm[0] + force[0] * arm[2],
                           - force[0] * arm[1] + force[1] * arm[0]])

    return moment


def site_of_action(force, moment):
    F = [[0, force[2], - force[1]], [- force[2], 0, force[0]], [force[1], - force[0], 0]]
    Ff = np.linalg.pinv(F)
    Mm = np.reshape(moment, (3, 1))
    X = np.dot(Ff, Mm)
    soa = np.ravel(X)
    return soa


def result_of_forces_and_moments(force1, site_of_action1,
                                 force2=np.array([0, 0, 0]), site_of_action2=np.array([0, 0, 0]),
                                 force3=np.array([0, 0, 0]), site_of_action3=np.array([0, 0, 0]),
                                 force4=np.array([0, 0, 0]), site_of_action4=np.array([0, 0, 0]),
                                 force5=np.array([0, 0, 0]), site_of_action5=np.array([0, 0, 0]),
                                 force6=np.array([0, 0, 0]), site_of_action6=np.array([0, 0, 0]),
                                 force7=np.array([0, 0, 0]), site_of_action7=np.array([0, 0, 0]),
                                 force8=np.array([0, 0, 0]), site_of_action8=np.array([0, 0, 0]),
                                 force9=np.array([0, 0, 0]), site_of_action9=np.array([0, 0, 0]),
                                 force10=np.array([0, 0, 0]), site_of_action10=np.array([0, 0, 0])):
    force = force1 + force2 + force3 + force4 + force5 + force6 + force7 + force8 + force9 + force10

    moment1 = moment_of_force(force1, site_of_action1)
    moment2 = moment_of_force(force2, site_of_action2)
    moment3 = moment_of_force(force3, site_of_action3)
    moment4 = moment_of_force(force4, site_of_action4)
    moment5 = moment_of_force(force5, site_of_action5)
    moment6 = moment_of_force(force6, site_of_action6)
    moment7 = moment_of_force(force7, site_of_action7)
    moment8 = moment_of_force(force8, site_of_action8)
    moment9 = moment_of_force(force9, site_of_action9)
    moment10 = moment_of_force(force10, site_of_action10)

    moment = moment1 + moment2 + moment3 + moment4 + moment5 + moment6 + moment7 + moment8 + moment9 + moment10

    soa = site_of_action(force, moment)

    return force, moment, soa


def ligament_force(lig0, lig1, lig2, klig):
    lig = lig2 - lig1
    lig_length = np.linalg.norm(lig)
    dlig_length = lig_length - lig0
    dlig = lig * dlig_length / lig_length

    if lig0 < lig_length:
        F_lig = - klig * dlig
    else:
        F_lig = np.array([0, 0, 0])
    return F_lig


def muscle_force(i):
    force = np.array([-muscle[i, 3], -muscle[i, 1], -muscle[i, 2]])
    moment = np.array([-muscle[i, 6], -muscle[i, 4], -muscle[i, 5]]) * 1000
    soa = site_of_action(force, moment)
    return force, moment, soa


def force_equilibrium4(i, fii):
    step_F = 0.001
    alpha = 1.1
    beta = 0.5
    m = 0

    F, M, _, _, _, _, _, _, _, _, _, _ = resultant_force(i)

    F_length = np.linalg.norm(F)
    M_length = np.linalg.norm(M)
    print("{:<3} {:<7} {:<15} {:<10} {:<20} {:<10} {:<20}".format(' ', ' ', step_F, 'F length =', F_length,
                                                                  'M length =', M_length))

    F_M = pd.DataFrame(np.append(i, [fii, m, F_length, M_length]).reshape(1, 5))
    F_M.to_csv(file1, index=False, mode='a', header=False)

    while True:
        if F_length < 1:
            break

        m += 1

        F_transform = np.array([[1, 0, 0, F[0] * step_F], [0, 1, 0, F[1] * step_F],
                                [0, 0, 1, F[2] * step_F], [0, 0, 0, 1]])
        flex.transform(F_transform, inplace=True)
        flex_cartilage.transform(F_transform, inplace=True)

        F_new, M_new, _, _, _, _, _, _, _, _, _, _ = resultant_force(i)

        F_length_new = np.linalg.norm(F_new)
        M_length_new = np.linalg.norm(M_new)

        F_M = pd.DataFrame(np.append(i, [fii, m, F_length_new, M_length_new]).reshape(1, 5))
        F_M.to_csv(file1, index=False, mode='a', header=False)

        if (F_length_new < 1) or (step_F < 1e-05):
            print("{:<3} {:<7} {:<15} {:<10} {:<20} {:<10} {:<20}".format(m, ' ', ' ', 'F length =', F_length_new,
                                                                          'M length =', M_length_new))
            break

        if 1 * F_length > F_length_new:
            step_F *= alpha
            step_F = round(step_F, 10)
            print("{:<3} {:<7} {:<15} {:<10} {:<20} {:<10} {:<20}".format(m, 'alpha', step_F, 'F length =',
                                                                          F_length_new, 'M length =', M_length_new))

        elif 1 * F_length < F_length_new:
            step_F *= beta
            step_F = round(step_F, 10)

            print("{:<3} {:<7} {:<15} {:<10} {:<20} {:<10} {:<20}".format(m, 'beta', step_F, 'F length =', F_length_new,
                                                                          'M length =', M_length_new))

        else:
            print("{:<3} {:<7} {:<15} {:<10} {:<20} {:<10} {:<20}".format(m, ' ', step_F, 'F length =', F_length_new,
                                                                          'M length =', M_length_new))

        F, F_length, M, M_length = F_new, F_length_new, M_new, M_length_new

    F, M, soa, N, soaN, stress, ACL, PCL, LCL, MCL, F_mus, soa_mus = resultant_force(i)

    return F, soa, N, soaN, ACL, PCL, LCL, MCL, stress, F_mus, soa_mus


def resultant_force(i):
    if model == 'full':
        ACL1 = np.array(tibia.points[2427])
        ACL2 = np.array(flex.points[8058])

        PCL1 = np.array(tibia.points[4428])
        PCL2 = np.array(flex.points[6507])

        MCL1 = np.array(tibia.points[5699])
        MCL2 = np.array(flex.points[9854])

        LCL1 = np.array([55, 20, -40])
        LCL2 = np.array(flex.points[8986])

    elif model == 'simple':
        ACL1 = np.array(tibia.points[366])
        ACL2 = np.array(flex.points[1107])

        PCL1 = np.array(tibia.points[452])
        PCL2 = np.array(flex.points[1568])

        MCL1 = np.array(tibia.points[1313])
        MCL2 = np.array(flex.points[1804])

        LCL1 = np.array([55, 20, -40])
        LCL2 = np.array(flex.points[146])

    F_ACL = ligament_force(ACL0, ACL1, ACL2, kACL)
    F_PCL = ligament_force(PCL0, PCL1, PCL2, kPCL)
    F_LCL = ligament_force(LCL0, LCL1, LCL2, kLCL)
    F_MCL = ligament_force(MCL0, MCL1, MCL2, kMCL)

    ACL = [F_ACL, ACL1, ACL2, 'ACL', 'ACL_force']
    PCL = [F_PCL, PCL1, PCL2, 'PCL', 'PCL_force']
    LCL = [F_LCL, LCL1, LCL2, 'LCL', 'LCL_force']
    MCL = [F_MCL, MCL1, MCL2, 'MCL', 'MCL_force']

    N, Nsoa, stress = normal_force()
    F_mus, M_mus, soa_mus = muscle_force(i)

    force, moment, soa = result_of_forces_and_moments(F_ACL, ACL2, F_PCL, PCL2, F_LCL, LCL2, F_MCL, MCL2,
                                                      N, Nsoa, F_mus, soa_mus)
    moment[0] = 0
    soa = site_of_action(force, moment)

    return force, moment, soa, N, Nsoa, stress, ACL, PCL, LCL, MCL, F_mus, soa_mus
