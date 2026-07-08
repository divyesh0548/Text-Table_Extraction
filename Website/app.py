import os
import datetime
import secrets
import camelot
from functools import wraps
from bson import ObjectId
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, get_jwt
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from threading import Thread
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=24)

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb+srv://divyesh95:div0548@cluster0.iy6tk.mongodb.net?retryWrites=true&w=majority&appName=Cluster0'  # Update with your MongoDB URI
app.config['MONGO_DB_NAME'] = 'table_extraction'

# Email Configuration (Update with your SMTP settings)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'divyeshparmar.svma@gmail.com'
app.config['MAIL_PASSWORD'] = 'kcwp hmch ohox pgna'   
app.config['MAIL_DEFAULT_SENDER'] = 'divyeshparmar.svma@gmail.com'

# Initialize extensions
jwt = JWTManager(app)
mail = Mail(app)

# MongoDB connection
try:
    client = MongoClient(app.config['MONGO_URI'])
    db = client[app.config['MONGO_DB_NAME']]

    # Test connection
    client.admin.command('ismaster')
    print("✅ Connected to MongoDB successfully!")

except Exception as e:
    print(f"❌ MongoDB connection failed: {str(e)}")
    print("Please make sure MongoDB is running on your system.")

# MongoDB Collections
users_collection = db.users
user_requests_collection = db.user_requests

# Blacklist for JWT tokens (in production, use Redis or database)
blacklisted_tokens = set()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload['jti'] in blacklisted_tokens

# Database Models (MongoDB Document Classes)
class User:
    def __init__(self, email, name, password=None, is_admin=False, is_approved=False, 
                 is_first_login=True, temporary_password=None, _id=None, created_at=None):
        self._id = _id or ObjectId()
        self.email = email
        self.name = name
        self.password_hash = None
        self.is_admin = is_admin
        self.is_approved = is_approved
        self.is_first_login = is_first_login
        self.temporary_password = temporary_password
        self.created_at = created_at or datetime.datetime.utcnow()

        if password:
            self.set_password(password)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def set_temporary_password(self, temp_password):
        self.temporary_password = generate_password_hash(temp_password)

    def check_temporary_password(self, temp_password):
        if not self.temporary_password:
            return False
        return check_password_hash(self.temporary_password, temp_password)

    def save(self):
        """Save user to MongoDB"""
        user_doc = {
            '_id': self._id,
            'email': self.email,
            'name': self.name,
            'password_hash': self.password_hash,
            'is_admin': self.is_admin,
            'is_approved': self.is_approved,
            'is_first_login': self.is_first_login,
            'temporary_password': self.temporary_password,
            'created_at': self.created_at
        }

        # Remove None values
        user_doc = {k: v for k, v in user_doc.items() if v is not None}

        result = users_collection.update_one(
            {'_id': self._id},
            {'$set': user_doc},
            upsert=True
        )
        return result

    @staticmethod
    def find_by_email(email):
        """Find user by email"""
        user_doc = users_collection.find_one({'email': email})
        if user_doc:
            return User(
                _id=user_doc['_id'],
                email=user_doc['email'],
                name=user_doc.get('name', ''),
                is_admin=user_doc.get('is_admin', False),
                is_approved=user_doc.get('is_approved', False),
                is_first_login=user_doc.get('is_first_login', True),
                temporary_password=user_doc.get('temporary_password'),
                created_at=user_doc.get('created_at')
            )._set_password_hash(user_doc.get('password_hash'))
        return None

    @staticmethod
    def find_by_id(user_id):
        """Find user by ObjectId"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        user_doc = users_collection.find_one({'_id': user_id})
        if user_doc:
            return User(
                _id=user_doc['_id'],
                email=user_doc['email'],
                name=user_doc.get('name', ''),
                is_admin=user_doc.get('is_admin', False),
                is_approved=user_doc.get('is_approved', False),
                is_first_login=user_doc.get('is_first_login', True),
                temporary_password=user_doc.get('temporary_password'),
                created_at=user_doc.get('created_at')
            )._set_password_hash(user_doc.get('password_hash'))
        return None

    def _set_password_hash(self, password_hash):
        """Internal method to set password hash"""
        self.password_hash = password_hash
        return self

class UserRequest:
    def __init__(self, email, name, status='pending', _id=None, created_at=None):
        self._id = _id or ObjectId()
        self.email = email
        self.name = name
        self.status = status
        self.created_at = created_at or datetime.datetime.utcnow()

    def save(self):
        """Save user request to MongoDB"""
        request_doc = {
            '_id': self._id,
            'email': self.email,
            'name': self.name,
            'status': self.status,
            'created_at': self.created_at
        }

        result = user_requests_collection.update_one(
            {'_id': self._id},
            {'$set': request_doc},
            upsert=True
        )
        return result

    @staticmethod
    def find_pending():
        """Find all pending requests"""
        pending_docs = user_requests_collection.find({'status': 'pending'}).sort('created_at', -1)
        requests = []
        for doc in pending_docs:
            req = UserRequest(
                _id=doc['_id'],
                email=doc['email'],
                name=doc.get('name', ''),
                status=doc['status'],
                created_at=doc.get('created_at')
            )
            requests.append(req)
        return requests

    @staticmethod
    def find_by_id(request_id):
        """Find request by ObjectId"""
        if isinstance(request_id, str):
            request_id = ObjectId(request_id)

        doc = user_requests_collection.find_one({'_id': request_id})
        if doc:
            req = UserRequest(
                _id=doc['_id'],
                email=doc['email'],
                name=doc.get('name', ''),
                status=doc['status'],
                created_at=doc.get('created_at')
            )
            return req
        return None

    @staticmethod
    def find_by_email_and_status(email, status):
        """Find request by email and status"""
        doc = user_requests_collection.find_one({'email': email, 'status': status})
        if doc:
            req = UserRequest(
                _id=doc['_id'],
                email=doc['email'],
                name=doc.get('name', ''),
                status=doc['status'],
                created_at=doc.get('created_at')
            )
            return req
        return None

# Utility Functions
def send_async_email(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            print(f"Error sending email: {str(e)}")

def send_email(subject, recipients, text_body, html_body):
    msg = Message(subject, recipients=recipients)
    msg.body = text_body
    msg.html = html_body
    Thread(target=send_async_email, args=(app, msg)).start()

def generate_temporary_password():
    return secrets.token_urlsafe(12)

def admin_required(f):
    @wraps(f)
    @jwt_required()
    def decorated_function(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)
        if not user or not user.is_admin:
            return jsonify({'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/user/dashboard',methods=['GET', 'POST'])
def user_dashboard():
    error = None
    table_num = None
    table_reports = []
    excel_ready = False
    pages = "all"
    pdf_filename = "converted"

    #Function to remove repeating headers
    def remove_duplicate_headers_largest_content(df):
        df['row_signature'] = df.apply(lambda row: tuple(str(cell).strip().lower() for cell in row if str(cell).strip()), axis=1)
        # Count non-empty cells per row
        df['non_empty_cells'] = df.apply(lambda row: sum(1 for cell in row if str(cell).strip()), axis=1)
        # Or use total character length of all cells combined as "size"
        df['content_length'] = df.apply(lambda row: sum(len(str(cell)) for cell in row), axis=1)
    
        # Find duplicated signatures
        duplicated_sigs = df['row_signature'][df['row_signature'].duplicated(keep=False)]
        duplicated_sigs_unique = duplicated_sigs.unique()
        
        rows_to_drop = []
        for sig in duplicated_sigs_unique:
            rows = df[df['row_signature'] == sig]
            # Select the row with maximum content length (size)
            idx_to_keep = rows['content_length'].idxmax()
            # Drop other duplicates except the largest (likely header)
            to_drop = rows.index.difference([idx_to_keep])
            rows_to_drop.extend(to_drop)
        
        df_cleaned = df.drop(rows_to_drop).drop(['row_signature', 'non_empty_cells', 'content_length'], axis=1)
        return df_cleaned.reset_index(drop=True)

    if request.method == 'POST'  and "download" not in request.form:
        selection = request.form.get('selection')
        pdf_file = request.files.get('pdf_file')
        pdf_filename = pdf_file.filename
        if not pdf_file or pdf_file.filename == '':
            error = "No pdf uploaded"
            
        else:
            #Creating a temporary file for extraction
            temp_path = "temp.pdf"
            pdf_file.save(temp_path)
            use_stream = request.form.get('use_stream')
            merge_tables = request.form.get('merge_tables')
            print("temp pdf saved")
            try:
                #Processing the pages input
                if selection == "range":
                    flavor = "stream" if use_stream else "lattice"
                    print(flavor)
                    print("Range selected")
                    page_range = request.form.get('page_range')
                    print(page_range)

                    start_end = page_range.replace(' ', '').split('-')
                    if len(start_end) != 2:
                         raise ValueError("Input should be in the format 'start-end', e.g., '2-5'")
                    start, end = map(int, start_end)
                    pages =  ','.join(str(i) for i in range(start, end+1))
                    print(f"Converted page range : {pages}")

                elif selection == "pages":
                    flavor = "stream" if use_stream else "lattice"
                    print("Pages selected")
                    pages = request.form.get('specific_pages')
                    page_numbers = [int(x) for x in pages.split(',')]
                    pages = ','.join(map(str, page_numbers))
                    print(f"Page numbers : {pages}")

                elif selection == "all":
                    flavor = "stream" if use_stream else "lattice"
                    pages = "all"
                    print("All pages selected selected")  
                
                #Reading tables using camelot
                tables = camelot.read_pdf(pdf_file, pages=pages, flavor=flavor) 
                print(f"{len(tables)} table(s) found")
                table_num = len(tables)

                #Printing tables to DataFrames
                # for index, tablePD in enumerate(tables):
                #     df = tablePD.df
                #     print(f"{index + 1} table into Dataframe: {df}")
                
                #Getting table extraction report
                for index,table in enumerate(tables):
                    report = table.parsing_report
                    table_reports.append({
                        'index' : index + 1,
                        'accuracy' : report.get('accuracy', 'N/A')
                    })

                #Merging tables if required
                if(merge_tables):
                    print("Merge tables selected")
                    all_tables_df = pd.concat([table.df for table in tables], ignore_index=True)
                    final_df = remove_duplicate_headers_largest_content(all_tables_df)
                    print(f"Final Dataframe : \n {final_df}")
                    final_df.to_excel('extracted_tables.xlsx', index=False, header=False)
                else:
                    print("Merge tables not selected")
                    tables.export('extracted_tables.xlsx', f='excel')
                    pass
 
                #Cleaning up temporary files
                if os.path.exists('temp.pdf'):
                    os.remove('temp.pdf')
                    print("temp pdf removed")

                print(f"pdf name : {pdf_filename}")
                # tables.export( pdf_filename.replace('.pdf', '.xlsx'), f='excel')
               
                if os.path.exists('extracted_tables.xlsx'):
                        excel_ready = True
                        print("Excel is ready to download")

                return render_template('user_dashboard.html', error=error, table_num=table_num,
                                       table_reports=table_reports, excel_ready=excel_ready)

            except Exception as e:
                error = "Error: " + str(e)

    if request.method == "POST" and "download" in request.form:
         return send_file("extracted_tables.xlsx", as_attachment=True)
        #  return send_file(pdf_filename.replace('.pdf', '.xlsx'), as_attachment=True)

    #Remove excel sheet 
    if os.path.exists('extracted_tables.xlsx'):
        os.remove('extracted_tables.xlsx')
        print("Extracted excelsheet removed")
    
    return render_template('user_dashboard.html', error=error, table_num=table_num, table_reports=table_reports, excel_ready=excel_ready)

@app.route('/change_password')
def change_password():
    return render_template('change_password.html')

# API Routes
@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')

        if not email or not name:
            return jsonify({'message': 'Email and name are required'}), 400

        # Check if user already exists
        if User.find_by_email(email):
            return jsonify({'message': 'User already exists'}), 400

        # Check if request already exists
        if UserRequest.find_by_email_and_status(email, 'pending'):
            return jsonify({'message': 'Registration request already submitted'}), 400

        # Create new user request (NO PASSWORD REQUIRED)
        user_request = UserRequest(email=email, name=name)
        user_request.save()

        return jsonify({'message': 'Registration request submitted successfully. Please wait for admin approval.'}), 201

    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        user = User.find_by_email(email)

        if not user:
            return jsonify({'message': 'Invalid credentials'}), 401

        if not user.is_approved:
            return jsonify({'message': 'Account not approved yet'}), 401

        # Check if it's first login with temporary password
        if user.is_first_login and user.temporary_password:
            if user.check_temporary_password(password):
                access_token = create_access_token(identity=str(user._id))
                return jsonify({
                    'access_token': access_token,
                    'is_admin': user.is_admin,
                    'is_first_login': True,
                    'message': 'Please change your password'
                }), 200

        # Regular login (after password has been changed)
        if user.check_password(password):
            access_token = create_access_token(identity=str(user._id))
            return jsonify({
                'access_token': access_token,
                'is_admin': user.is_admin,
                'is_first_login': False
            }), 200

        return jsonify({'message': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/change_password', methods=['POST'])
@jwt_required()
def api_change_password():
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)

        if not user:
            return jsonify({'message': 'User not found'}), 404

        data = request.get_json()
        new_password = data.get('new_password')

        if not new_password:
            return jsonify({'message': 'New password is required'}), 400

        # Set the new password as the main password
        user.set_password(new_password)
        user.is_first_login = False
        user.temporary_password = None  # Clear temporary password

        user.save()

        return jsonify({'message': 'Password changed successfully'}), 200

    except Exception as e:
        print(f"Change password error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/logout', methods=['POST'])
@jwt_required()
def api_logout():
    try:
        token = get_jwt()
        jti = token['jti']
        blacklisted_tokens.add(jti)
        return jsonify({'message': 'Successfully logged out'}), 200
    except Exception as e:
        print(f"Logout error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    try:
        # Count pending user requests
        pending_count = user_requests_collection.count_documents({'status': 'pending'})

        # Count approved users created today
        today = datetime.datetime.utcnow().date()
        start_of_day = datetime.datetime.combine(today, datetime.time.min)
        end_of_day = datetime.datetime.combine(today, datetime.time.max)

        approved_today_count = users_collection.count_documents({
            'is_approved': True,
            'created_at': {'$gte': start_of_day, '$lte': end_of_day}
        })

        # Count total users
        total_users_count = users_collection.count_documents({ "is_admin": { "$ne": True } })

        return jsonify({
            'pending_requests': pending_count,
            'approved_today': approved_today_count,
            'total_users': total_users_count
        }), 200
    except Exception as e:
        print(f"Error fetching admin stats: {e}")
        return jsonify({'message': 'Error fetching stats'}), 500

@app.route('/api/admin/requests', methods=['GET'])
@admin_required
def api_get_requests():
    try:
        requests = UserRequest.find_pending()
        requests_data = []
        for req in requests:
            requests_data.append({
                'id': str(req._id),
                'email': req.email,
                'name': req.name,
                'created_at': req.created_at.strftime('%Y-%m-%d %H:%M:%S') if req.created_at else ''
            })
        return jsonify({'requests': requests_data}), 200
    except Exception as e:
        print(f"Get requests error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/admin/approve_request/<request_id>', methods=['POST'])
@admin_required
def api_approve_request(request_id):
    try:
        user_request = UserRequest.find_by_id(request_id)

        if not user_request:
            return jsonify({'message': 'Request not found'}), 404

        if user_request.status != 'pending':
            return jsonify({'message': 'Request already processed'}), 400

        # Generate temporary password
        temp_password = generate_temporary_password()

        # Create new user with ONLY temporary password (no main password yet)
        new_user = User(
            email=user_request.email,
            name=user_request.name,
            is_approved=True,
            is_first_login=True
        )
        new_user.set_temporary_password(temp_password)

        # Update request status
        user_request.status = 'approved'

        # Save both documents
        new_user.save()
        user_request.save()

        # Send email with temporary password
        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #2c5aa0;">Account Approved! 🎉</h2>
            <p>Dear {user_request.name},</p>
            <p>Your registration request has been approved! You can now access the system.</p>

            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #28a745;">Login Credentials:</h3>
                <p><strong>Email:</strong> {user_request.email}</p>
                <p><strong>Temporary Password:</strong> <code style="background-color: #e9ecef; padding: 4px 8px; border-radius: 3px;">{temp_password}</code></p>
            </div>

            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>⚠️ Important:</strong> You will be required to change this temporary password on your first login for security purposes.</p>
            </div>

            <p>Please login at your earliest convenience and set up your permanent password.</p>
            <p>If you have any questions, please contact our support team.</p>

            <hr style="margin: 30px 0;">
            <p style="color: #6c757d; font-size: 14px;">This is an automated message. Please do not reply to this email.</p>
        </div>
        """

        send_email(
            subject='🎉 Account Approved - Welcome!',
            recipients=[user_request.email],
            text_body=f'Hello {user_request.name},\n\nYour account has been approved!\n\nLogin Details:\nEmail: {user_request.email}\nTemporary Password: {temp_password}\n\nPlease login and change your password immediately.\n\nThank you!',
            html_body=html_body
        )

        return jsonify({'message': 'Request approved and email sent to user'}), 200

    except Exception as e:
        print(f"Approve request error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/admin/reject_request/<request_id>', methods=['POST'])
@admin_required
def api_reject_request(request_id):
    try:
        user_request = UserRequest.find_by_id(request_id)

        if not user_request:
            return jsonify({'message': 'Request not found'}), 404

        if user_request.status != 'pending':
            return jsonify({'message': 'Request already processed'}), 400

        user_request.status = 'rejected'
        user_request.save()

        # Send rejection email
        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #dc3545;">Registration Request Update</h2>
            <p>Dear {user_request.name},</p>
            <p>Thank you for your interest in our platform. Unfortunately, your registration request has been declined at this time.</p>
            <p>If you believe this is an error or would like to discuss this decision, please contact our support team.</p>
            <p>Thank you for your understanding.</p>

            <hr style="margin: 30px 0;">
            <p style="color: #6c757d; font-size: 14px;">This is an automated message. Please do not reply to this email.</p>
        </div>
        """

        send_email(
            subject='Registration Request Update',
            recipients=[user_request.email],
            text_body=f'Hello {user_request.name},\n\nYour registration request has been declined. Please contact support if you have questions.\n\nThank you.',
            html_body=html_body
        )

        return jsonify({'message': 'Request rejected and email sent to user'}), 200

    except Exception as e:
        print(f"Reject request error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def api_get_profile():
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)

        print("User profile status from API:", user)

        if not user:
            return jsonify({'message': 'User not found'}), 404

        return jsonify({
            'email': user.email,
            'name': user.name,
            'is_admin': user.is_admin,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else ''
        }), 200

    except Exception as e:
        print(f"Get profile error: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500

def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Create indexes for users collection
        users_collection.create_index("email", unique=True)
        users_collection.create_index("is_admin")
        users_collection.create_index("is_approved")

        # Create indexes for user_requests collection
        user_requests_collection.create_index("email")
        user_requests_collection.create_index("status")
        user_requests_collection.create_index("created_at")

        print("✅ Database indexes created successfully")

    except Exception as e:
        print(f"⚠️  Error creating indexes: {str(e)}")

def create_admin_user():
    """Create default admin user if doesn't exist"""
    try:
        admin_email = 'admin@example.com'
        admin = User.find_by_email(admin_email)

        if not admin:
            admin = User(
                email=admin_email,
                name='System Administrator',
                password='admin123',  # Admin gets a direct password, no temporary password flow
                is_admin=True,
                is_approved=True,
                is_first_login=False
            )
            admin.save()
            print("✅ Default admin user created: admin@example.com / admin123")
        else:
            print("✅ Admin user already exists")

    except Exception as e:
        print(f"⚠️  Error creating admin user: {str(e)}")

if __name__ == '__main__':
    try:
        # Initialize database
        create_indexes()
        create_admin_user()

        print("\n" + "="*60)
        print("🚀 FLASK USER MANAGEMENT SYSTEM - CORRECTED FLOW!")
        print("="*60)
        print("📧 Admin: admin@example.com / admin123")
        print("🌐 URL: http://localhost:5000")
        print("📦 MongoDB Database: user_management")
        print("\n✅ CORRECTED REGISTRATION FLOW:")
        print("1. User registers with EMAIL + NAME only (no password)")
        print("2. Admin approves request")
        print("3. User gets temporary password via email")
        print("4. User logs in and changes password")
        print("5. User can access main content")
        print("="*60)

        app.run(debug=True)

    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Please ensure MongoDB is running on your system.")
