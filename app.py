from flask import Flask, render_template, request, session
import numpy as np
# Import your newly written algorithm math file
from train_model import CapstonePredictorEngine

app = Flask(__name__)
app.secret_key = 'your-capstone-secret-key-keep-it-safe'

# Initialize the mathematical engine using your dataset csv
engine = CapstonePredictorEngine('dataset/philippine_gwa_dataset.csv')

def normalize_gpa(gpa, grading_system):
    """Normalize user input to match the 1.0 - 5.0 Inverse scale of your CSV"""
    try:
        val = float(gpa)
        if grading_system == 'inverse': return val
        elif grading_system == 'us_gpa': return max(1.0, min(5.0, 5.0 - val))
        elif grading_system == 'percentage': return max(1.0, min(5.0, 1.0 + ((100.0 - val) * 0.08)))
        return val
    except Exception:
        return 3.0

@app.route('/')
def index(): return render_template('about.html', total_students=6000, predictions=2027, accuracy=92, programs=4)

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', total_students=6000, placed_students=2027, avg_salary='28,500', placement_rate=85, recent_predictions=[])

@app.route('/predictor', methods=['GET'])
def predictor(): return render_template('predictor.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract frontend elements
        grading_system = request.form.get('grading_system')
        gpa = request.form.get('gpa')
        course_type = request.form.get('course_type')
        skills = request.form.getlist('skills')
        
        internships = int(request.form.get('internships', 0))
        projects = int(request.form.get('projects', 0))
        backlogs = int(request.form.get('backlogs', 0))
        resume_quality = int(request.form.get('resume_quality', 5))
        communication_rate = int(request.form.get('communication_rate', 5))
        coding_score = int(request.form.get('coding_score', 50))
        aptitude_score = int(request.form.get('aptitude_score', 50))
        
        gpa_normalized = normalize_gpa(gpa, grading_system)
        num_skills = len(skills)
        
        # Map values exactly onto your dataset column shapes
        python_skill = 1.0 if 'Python' in skills else 0.0
        dsa_skill = 1.0 if 'Data Structures and Algorithms' in skills else 0.0
        ml_skill = 1.0 if 'Machine Learning' in skills else 0.0
        web_dev_skill = 1.0 if 'Web Development' in skills else 0.0
        skill_score = num_skills / 4.0
        
        coding_scaled = coding_score / 100.0
        comm_scaled = communication_rate / 10.0
        aptitude_scaled = aptitude_score / 100.0
        resume_scaled = resume_quality / 10.0
        internships_scaled = min(internships, 5) / 5.0
        projects_scaled = min(projects, 5) / 5.0
        backlogs_scaled = min(backlogs, 10) / 10.0
        
        program_it = 1.0 if course_type == 'BS Information Technology' else 0.0
        program_cs = 1.0 if course_type == 'BS Computer Science' else 0.0
        program_is = 1.0 if course_type == 'BS Information Systems' else 0.0
        program_cpe = 1.0 if course_type == 'BS Computer Engineering' else 0.0

      # --- FIX: Scale and structure all 17 features to match your decimal-based dataset ---
        coding_scaled = coding_score / 100.0         # 10 becomes 0.10
        comm_scaled = communication_rate / 10.0      # 4 becomes 0.40
        aptitude_scaled = aptitude_score / 100.0    # 78 becomes 0.78
        resume_scaled = resume_quality / 100.0      # 50 becomes 0.50 (Normalized to 0.0 - 1.0)
        
        # Max caps ensure inputs scale between 0.0 and 1.0 if someone enters large numbers
        internships_scaled = min(internships, 5) / 5.0
        projects_scaled = min(projects, 5) / 5.0
        backlogs_scaled = min(backlogs, 10) / 10.0

        features_array = np.array([[
            python_skill,          # 1. python_skill (0.0 or 1.0)
            dsa_skill,             # 2. dsa_skill (0.0 or 1.0)
            ml_skill,              # 3. ml_skill (0.0 or 1.0)
            web_dev_skill,         # 4. web_dev_skill (0.0 or 1.0)
            coding_scaled,         # 5. coding_score (Decimal, e.g., 0.10)
            comm_scaled,           # 6. communication_score (Decimal, e.g., 0.40)
            aptitude_scaled,       # 7. aptitude_score (Decimal, e.g., 0.78)
            internships_scaled,    # 8. internships (Decimal fraction)
            projects_scaled,       # 9. projects (Decimal fraction)
            backlogs_scaled,       # 10. backlogs (Decimal fraction)
            resume_scaled,         # 11. resume_score (Decimal fraction)
            skill_score,           # 12. skill_score (0.50)
            program_it,            # 13. Program_IT (1.0)
            program_cs,            # 14. Program_CS (0.0)
            program_is,            # 15. Program_IS (0.0)
            program_cpe,           # 16. Program_CPE (0.0)
            gpa_normalized         # 17. GWA (Decimal-based normalized GPA)
        ]])
        
        # Save session data for student diagnostic graphs
        session['student_data'] = {
            'grading_system': grading_system, 'gpa_original': gpa, 'gpa_normalized': round(gpa_normalized, 2),
            'course_type': course_type, 'skills': skills, 'num_skills': num_skills, 'internships': internships,
            'projects': projects, 'backlogs': backlogs, 'resume_quality': resume_quality,
            'communication_rate': communication_rate, 'coding_score': coding_score, 'aptitude_score': aptitude_score
        }
        
        # --- EXECUTE RETRIEVAL OR FALLBACK MATH ALGORITHMS ---
        
        # 1. Try to fetch an exact historical match from the dataset first
        exact_placed, exact_salary = engine.find_exact_match(features_array)
        
        if exact_placed is not None:
            # EXACT MATCH FOUND: Copy results directly from dataset!
            is_placed = exact_placed
            placement_probability = 100.0 if is_placed else 0.0
            predicted_base = exact_salary
            
            prediction = {
                'placement_status': 'Placed' if is_placed else 'Unplaced',
                'placement_probability': placement_probability,
                'is_exact_match': True # Optional flag for your UI to show it's a real record
            }
        else:
            # NO EXACT MATCH: Fall back to your predictive algorithms
            placement_probability = engine.predict_placement_knn(features_array, k=5)
            rf_score = engine.predict_placement_random_forest_mimic(features_array)
            is_placed = (placement_probability >= 50.0) and (rf_score >= 50.0)
            
            prediction = {
                'placement_status': 'Placed' if is_placed else 'Unplaced',
                'placement_probability': round(placement_probability, 1),
                'is_exact_match': False
            }
            
            if is_placed:
                predicted_base = engine.predict_salary_linear_regression(features_array)
            else:
                predicted_base = 0
                
        # --- CALCULATE JOB ROLE VIA DYNAMIC TRAIT WEIGHTING (ONLY IF PLACED) ---
        if is_placed:
            prediction['placement_message'] = f'Congratulations! You have a {prediction["placement_probability"]}% probability of getting placed.'
            predicted_base = engine.predict_salary_linear_regression(features_array)
            
            # 1. Math-based rule weights to determine the matching Job Role
            # We look at specific skills, course types, coding scores, and aptitude
            se_score = (python_skill * 30) + (dsa_skill * 30) + (coding_scaled * 40) + (program_cs * 10)
            web_score = (web_dev_skill * 45) + (coding_scaled * 35) + (comm_scaled * 10) + (program_it * 10)
            ds_score = (python_skill * 35) + (ml_skill * 45) + (aptitude_scaled * 20) + (program_cs * 10)
            analyst_score = (comm_scaled * 30) + (aptitude_scaled * 35) + (python_skill * 15) + (program_is * 20) + (program_it * 10)

            # Find the highest classification score matching the traits
            scores = {
                'Software Engineer': se_score,
                'Web Developer': web_score,
                'Data Scientist': ds_score,
                'Analyst': analyst_score
            }
            assigned_role = max(scores, key=scores.get)
            prediction['job_role'] = assigned_role

            prediction.update({
                'salary_mnc': f'{int(predicted_base * 1.2):,}',
                'salary_midsize': f'{int(predicted_base * 1.0):,}',
                'salary_startup': f'{int(predicted_base * 0.85):,}',
                'salary_toptech': f'{int(predicted_base * 1.5):,}'
            })
        else:
            prediction['placement_message'] = f'Your current placement probability is {prediction["placement_probability"]}%. Profile adjustments are required.'
            prediction['job_role'] = None # Clear role if unplaced
        
        # Simple checks to populate the UI warnings page
        weaknesses = []
        if gpa_normalized > 2.8:
            weaknesses.append({'area': 'Academic Performance (GWA)', 'message': 'High numeric inverse GWA limits profile competitiveness.', 'tips': ['Focus heavily on your core major subjects.']})
        if backlogs > 0:
            weaknesses.append({'area': 'Academic Backlogs', 'message': f'You have {backlogs} outstanding module(s).', 'tips': ['Clear remaining backlogs to pass placement filters.']})
        if coding_score < 60:
            weaknesses.append({'area': 'Coding Skills', 'message': 'Coding speed metric falls below key benchmarks.', 'tips': ['Practice algorithm tasks daily.']})
            
        prediction['weaknesses'] = weaknesses
        return render_template('predictor.html', prediction=prediction)
        
    except Exception as e:
        print(f"Error encountered in app engine loop: {str(e)}")
        return render_template('predictor.html', prediction=None)
        
        # 3. Use live Multiple Linear Regression to calculate salaries
        if is_placed:
            prediction['placement_message'] = f'Congratulations! You have a {prediction["placement_probability"]}% probability of getting placed.'
            predicted_base = engine.predict_salary_linear_regression(features_array)
            
            prediction.update({
                'salary_mnc': f'{int(predicted_base * 1.2):,}',
                'salary_midsize': f'{int(predicted_base * 1.0):,}',
                'salary_startup': f'{int(predicted_base * 0.85):,}',
                'salary_toptech': f'{int(predicted_base * 1.5):,}'
            })
        else:
            prediction['placement_message'] = f'Your current placement probability is {prediction["placement_probability"]}%. Profile adjustments are required.'
        
        # Simple checks to populate the UI warnings page
        weaknesses = []
        if gpa_normalized > 2.8:
            weaknesses.append({'area': 'Academic Performance (GWA)', 'message': 'High numeric inverse GWA limits profile competitiveness.', 'tips': ['Focus heavily on your core major subjects.']})
        if backlogs > 0:
            weaknesses.append({'area': 'Academic Backlogs', 'message': f'You have {backlogs} outstanding module(s).', 'tips': ['Clear remaining backlogs to pass placement filters.']})
        if coding_score < 60:
            weaknesses.append({'area': 'Coding Skills', 'message': 'Coding speed metric falls below key benchmarks.', 'tips': ['Practice algorithm tasks daily.']})
            
        prediction['weaknesses'] = weaknesses
        return render_template('predictor.html', prediction=prediction)
        
    except Exception as e:
        print(f"Error encountered in app engine loop: {str(e)}")
        return render_template('predictor.html', prediction=None)

@app.route('/skillsanalysis')
def skillsanalysis():
    return render_template('skills.html', student_data=session.get('student_data'))

if __name__ == '__main__':
    app.run(debug=True)