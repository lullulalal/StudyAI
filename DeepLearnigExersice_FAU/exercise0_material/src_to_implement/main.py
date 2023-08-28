import pattern
import generator

checker = pattern.Checker(100, 25)
checker.draw()
checker.show()

circle = pattern.Circle(20, 4, (5, 10))
circle.draw()
circle.show()

spectrum = pattern.Spectrum(1000)
spectrum.draw()
spectrum.show()

file_path = './exercise_data/'
label_path = './Labels.json'

gen = generator.ImageGenerator(file_path, label_path, 5, [32, 32, 3], rotation=True, mirroring=False, shuffle=False)
gen.show()