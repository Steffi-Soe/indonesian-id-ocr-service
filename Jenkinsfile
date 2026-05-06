pipeline {
    agent { label 'docker-apps2' }

    environment {
        APP_DIR = "/apps/acs/ocr-id/ocr-id"
        REPO = "https://github.com/Steffi-Soe/indonesian-id-ocr-service.git"
    }

    stages {

        stage('Update Source Code') {
            steps {
                sh """
                if [ ! -d "$APP_DIR/.git" ]; then
                    git clone $REPO $APP_DIR
                else
                    cd $APP_DIR
                    git pull
                fi
                """
            }
        }

        stage('Deploy Container') {
            steps {
                sh """
                cd $APP_DIR
                docker compose up -d --build
                """
            }
        }

    }
}