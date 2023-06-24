import forces as f
from setup import *
import time as tim


def initialization():
    start = tim.time()

    global fii
    fii = motion[0, 1]

    global fi_step
    fi_step = fii

    print('i =', 0)
    print('fi =', fii)

    F, soa, N, soaN, ACL, PCL, LCL, MCL, stress, F_mus, soa_mus = rolling_with_ligament(0, 0)
    flex_plot(fii, 0, F, soa, N, soaN, ACL, PCL, LCL, MCL, F_mus, soa_mus)

    stress_plot(fii, 0, stress)

    end = tim.time()
    time = end - start
    print('time =', time)
    x = np.array(position())
    print('position =', x, '\n\n')


def update_scene(i):
    start = tim.time()

    global fii
    fii = motion[(i + 1)*step, 1]

    global fi_step
    fi_step = fii - motion[i*step, 1]

    p1_1 = np.array(flex.points[14])
    p2_1 = np.array(flex.points[25])

    print('i =', i+1)
    print('fi =', fii)

    F, soa, N, soaN, ACL, PCL, LCL, MCL, stress, F_mus, soa_mus = rolling_with_ligament(i, fii)

    p1_2 = np.array(flex.points[14])
    p2_2 = np.array(flex.points[25])

    axis = actual_axis_of_rotation(p1_1, p2_1, p1_2, p2_2)

    flex_plot(fii, i, F, soa, N, soaN, ACL, PCL, LCL, MCL, F_mus, soa_mus, axis)
    stress_plot(fii, i, stress)

    end = tim.time()
    time = end - start
    print('time =', time)
    x = np.array(position())
    print('position =', x, '\n', '\n')


def rolling_with_ligament(i, fii):
    pm, pl = before_rotation()
    flex.rotate_vector(vector=pm - pl, angle=-fi_step, point=pm, inplace=True)
    flex_cartilage.rotate_vector(vector=pm - pl, angle=-fi_step, point=pm, inplace=True)
    F, soa, N, soaN, ACL, PCL, LCL, MCL, stress, F_mus, soa_mus = f.force_equilibrium4(i, fii)
    return F, soa, N, soaN, ACL, PCL, LCL, MCL, stress, F_mus, soa_mus


def before_rotation():
    cor4 = open(file0, 'a')

    a = 0

    while True:
        z = 0
        fiy = 0
        while True:
            collision, ncol = (tibia + tibial_cartilage).collision(flex + flex_cartilage)

            if ncol == 0:
                print('z')
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -z_step], [0, 0, 0, 1]])
                flex.transform(transform)
                flex_cartilage.transform(transform)
                z -= z_step
            else:
                break

        bodies, points_medial, points_lateral = contact_volume()

        if (not points_medial.any()) and (a == 1):
            print('medial, z')
            transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -z_step], [0, 0, 0, 1]])
            flex.transform(transform)
            flex_cartilage.transform(transform)
            z -= z_step
            a = 0
        elif (not points_medial.any()) and (a != 1):
            print('medial')
            flex.rotate_y(- fiy_step, inplace=True)
            flex_cartilage.rotate_y(- fiy_step, inplace=True)
            fiy -= fiy_step
            a = -1
        elif (not points_lateral.any()) and (a == -1):
            print('lateral, z')
            transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -z_step], [0, 0, 0, 1]])
            flex.transform(transform)
            flex_cartilage.transform(transform)
            z -= z_step
            a = 0
        elif (not points_lateral.any()) and (a != -1):
            print('lateral')
            flex.rotate_y(fiy_step, inplace=True)
            flex_cartilage.rotate_y(fiy_step, inplace=True)
            fiy += fiy_step
            a = 1
        else:
            break

    print('dz=', z, ', fiy=', fiy)

    cor4.write(str('dz = '))
    cor4.write(str(z))
    cor4.write(str(', dfiy = '))
    cor4.write(str(fiy))
    cor4.write('\n')
    cor4.close()

    pm = np.mean(points_medial, axis=0)
    pl = np.mean(points_lateral, axis=0)

    return pm, pl


def contact_volume():
    contact_volumes = (tibia + tibial_cartilage).boolean_intersection(flex + flex_cartilage)
    threshed = contact_volumes.threshold(0.001, invert=True)
    bodies = threshed.split_bodies()

    points_medial = np.empty((0, 3), int)
    points_lateral = np.empty((0, 3), int)

    for i in bodies.keys():
        point = np.array(bodies[i].center)
        if point[0] < 0:
            points_medial = np.append(points_medial, [np.array(bodies[i].center)], axis=0)
        else:
            points_lateral = np.append(points_lateral, [np.array(bodies[i].center)], axis=0)
    return contact_volumes, points_medial, points_lateral


def actual_axis_of_rotation(p1_1, p2_1, p1_2, p2_2):
    p1 = np.mean([p1_1, p1_2], axis=0)
    p2 = np.mean([p2_1, p2_2], axis=0)

    n1 = p1_1 - p1_2
    n2 = p2_1 - p2_2

    d1 = -n1 @ p1
    d2 = -n2 @ p2

    a = (n2[0] * n1[2] - n1[0] * n2[2]) / (n1[1] * n2[2] - n2[1] * n1[2])
    b = (n1[2] * d2 - n2[2] * d1) / (n1[1] * n2[2] - n2[1] * n1[2])

    x1 = 30
    y1 = a * x1 + b
    z1 = ((- n1[0] - a * n1[1]) * x1 - b * n1[1] - d1) / n1[2]

    x2 = -30
    y2 = a * x2 + b
    z2 = ((- n1[0] - a * n1[1]) * x2 - b * n1[1] - d1) / n1[2]

    pA = np.array([x1, y1, z1])
    pB = np.array([x2, y2, z2])
    axis = [pA, pB]

    axis_of_rotation = pd.DataFrame(np.append(pA, pB).reshape(1, 6))
    axis_of_rotation.to_csv(file, index=False, mode='a', header=False)

    return axis


def position():
    flex_position = flex.center
    x0 = flex_position[0]
    y0 = flex_position[1]
    z0 = flex_position[2]
    return x0, y0, z0


def angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    ang = np.arccos(dot_product) * 180 / math.pi
    return ang


def plot_lig(p2, lig):
    F_lig = lig[0]
    lig1 = lig[1]
    lig2 = lig[2]
    lig_name = lig[3]
    lig_force_name = lig[4]

    p2.add_mesh(pv.Line(lig1, lig2), color='springgreen', line_width=2, name=lig_name)
    F_lig_direction = F_lig / np.linalg.norm(F_lig)
    line = pv.Line(lig2, lig2 - F_lig)
    tip = pv.Cone(center=lig2 - F_lig_direction * 5, direction=F_lig_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='seagreen', line_width=5, name=lig_force_name)


def flex_plot(fi_act, time, F, soa, N, soaN, ACL, PCL, LCL, MCL, F_mus, soa_mus, axis=None):
    if axis is None:
        axis = [np.array([0, 0, 0]), np.array([0, 0, 0])]

    flex_pic = 'flex' + str(time) + '.png'
    text1 = 'time = ' + str(time)
    text2 = 'fi = ' + str(fi_act) + '°'

    p2 = pv.Plotter(off_screen=True, shape=(1, 2))
    p2.background_color = 'white'

    p2.subplot(0, 1)

    # p2.add_text(text1, font_size=35, position='upper_left')

    p2.add_mesh(flex, style='wireframe', color='linen')
    p2.add_mesh(tibia, style='wireframe', color='linen')
    p2.add_mesh(femoral_cartilage, style='wireframe', color='gold')
    p2.add_mesh(tibial_cartilage, style='wireframe', color='gold')

    line = pv.Line(axis[0], axis[1])
    p2.add_mesh(line, color='darkslateblue', line_width=2, name='axis')

    F_direction = -F / np.linalg.norm(F)
    line = pv.Line(soa, soa + F)
    tip = pv.Cone(center=soa - F_direction * 5, direction=F_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='darkred', line_width=5, name='force')

    N_direction = N / np.linalg.norm(N)
    line = pv.Line(soaN, soaN - N)
    tip = pv.Cone(center=soaN - N_direction * 5, direction=N_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='goldenrod', line_width=5, name='N')

    F_mus_direction = F_mus / np.linalg.norm(F_mus)
    line = pv.Line(soa_mus, soa_mus - F_mus)
    tip = pv.Cone(center=soa_mus - F_mus_direction * 5, direction=F_mus_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='coral', line_width=5, name='F_mus')

    plot_lig(p2, ACL)
    plot_lig(p2, PCL)
    plot_lig(p2, LCL)
    plot_lig(p2, MCL)

    p2.camera.position = (-500, 0, 10)
    p2.camera.focal_point = (100, 0, 10)

    p2.subplot(0, 0)

    # p2.add_text(text2, font_size=35, position='upper_edge')

    p2.add_mesh(flex, style='wireframe', color='linen')
    p2.add_mesh(tibia, style='wireframe', color='linen')
    p2.add_mesh(femoral_cartilage, style='wireframe', color='gold')
    p2.add_mesh(tibial_cartilage, style='wireframe', color='gold')

    line = pv.Line(axis[0], axis[1])
    p2.add_mesh(line, color='darkslateblue', line_width=2, name='axis')

    F_direction = -F / np.linalg.norm(F)
    line = pv.Line(soa, soa + F)
    tip = pv.Cone(center=soa - F_direction * 5, direction=F_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='darkred', line_width=5, name='force')

    N_direction = N / np.linalg.norm(N)
    line = pv.Line(soaN, soaN - N)
    tip = pv.Cone(center=soaN - N_direction * 5, direction=N_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='goldenrod', line_width=5, name='N')

    F_mus_direction = F_mus / np.linalg.norm(F_mus)
    line = pv.Line(soa_mus, soa_mus - F_mus)
    tip = pv.Cone(center=soa_mus - F_mus_direction * 5, direction=F_mus_direction, height=10, radius=2)
    p2.add_mesh(line + tip, color='coral', line_width=5, name='F_mus')

    plot_lig(p2, ACL)
    plot_lig(p2, PCL)
    plot_lig(p2, LCL)
    plot_lig(p2, MCL)

    p2.camera.position = (0, -500, 10)
    p2.camera.focal_point = (0, 100, 10)

    p2.show(screenshot=flex_pic, window_size=(2000, 2400), title=(text1+text2))


def stress_plot(fi_act, time, stress):
    file2 = 'stress' + str(time) + '.csv'
    picture = 'stress' + str(time) + '.png'
    text1 = 'time = ' + str(time)
    text2 = 'fi = ' + str(fi_act) + '°'

    if os.path.exists(file2):
        os.remove(file2)

    df_stress = pd.DataFrame(stress)
    df_stress.to_csv(file2, index=False, mode='a', header=False)

    mesh1 = flex.extract_cells(range(flex.n_cells))
    mesh1.cell_data['Normal stress [MPa]'] = stress

    p1 = pv.Plotter(off_screen=True)

    p1.add_text(text1, position='upper_left')
    p1.add_text(text2, position='upper_edge')
    p1.add_mesh(mesh1, scalars='Normal stress [MPa]', clim=[0.00000001, 10], below_color='grey', above_color='red',
                reset_camera=True)
    p1.view_yx()
    p1.set_viewup([0, -1, 0])
    p1.show(screenshot=picture)
